import numpy as np
from scipy.optimize import minimize, differential_evolution
from copy import deepcopy
import random
from ag_rewards.ag_func_correlations import calculate_correlation
from ag_objects.ag_obj_interpretation import create_segments, get_shift_by_idx
import matplotlib.pyplot as plt

# Пытаемся импортировать настройки из основного файла
try:
    from ag_runners.ag__run_interpr_from_landing import OPTIMIZATION_METHOD
except ImportError:
    # Если не удалось, используем значение по умолчанию
    OPTIMIZATION_METHOD = "Nelder-Mead"


class OptimizationVisualizer:
    """
    Класс для визуализации процесса оптимизации в реальном времени
    """

    def __init__(self, well, typewell, segments, max_iterations=50, manual_interpretation=None,
                 well_manual_interpretation=None):
        self.well = well
        self.typewell = typewell
        self.segments = segments
        self.manual_interpretation = manual_interpretation
        self.manual_well = well_manual_interpretation
        self.iterations = 0
        self.max_iterations = max_iterations
        self.history = {
            'objective_values': [],
            'shifts': [],
            'mse': [],
            'pearson': [],
            'segment_angles': []
        }
        self.angle_annotations = []

        # Добавляем флаг для заморозки оптимизации
        self.paused = False
        self.should_stop = False  # Флаг для полной остановки оптимизации

        self.setup_visualization()

    def setup_visualization(self):
        """Настройка окна визуализации"""
        plt.ion()  # Включение интерактивного режима
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Optimization Progress Visualization', fontsize=14)

        # Оси для отображения кривых
        self.axes[0, 0].set_title('Well Curves')
        self.axes[0, 0].set_xlabel('MD')
        self.axes[0, 0].set_ylabel('Value')

        # Оси для отображения целевой функции
        self.axes[0, 1].set_title('Objective Function Values')
        self.axes[0, 1].set_xlabel('Iterations')
        self.axes[0, 1].set_ylabel('Value')

        # Оси для отображения сегментов и их сдвигов
        self.axes[1, 0].set_title('Segment Shifts')
        self.axes[1, 0].set_xlabel('Segment VS')
        self.axes[1, 0].set_ylabel('Shift Value')

        # Оси для отображения траектории
        self.axes[1, 1].set_title('Well Trajectory')
        self.axes[1, 1].set_xlabel('MD')
        self.axes[1, 1].set_ylabel('TVT')

        # Инициализация линий для графиков
        self.lines = {}
        self.lines['well_curve'], = self.axes[0, 0].plot([], [], label='Well Curve', color='r')
        self.lines['synth_curve'], = self.axes[0, 0].plot([], [], label='Synthetic Curve', color='b')
        self.lines['manual_synth_curve'], = self.axes[0, 0].plot([], [], label='Manual Synth Curve', color='k',
                                                                 linestyle='--')

        self.lines['objective'], = self.axes[0, 1].plot([], [], label='Objective Value', color='g')
        self.lines['shifts'], = self.axes[1, 0].plot([], [], 'o-', label='Shifts', color='m')
        self.lines['manual_shifts'], = self.axes[1, 0].plot([], [], 'o--', label='Manual Shifts', color='k')

        self.lines['trajectory'], = self.axes[1, 1].plot([], [], label='Trajectory', color='k')
        self.lines['tvt'], = self.axes[1, 1].plot([], [], label='TVT', color='b')
        self.lines['manual_tvt'], = self.axes[1, 1].plot([], [], label='Manual TVT', color='g', linestyle='--')

        # Добавление кнопок для управления процессом оптимизации
        pause_ax = plt.axes([0.7, 0.01, 0.1, 0.05])
        stop_ax = plt.axes([0.81, 0.01, 0.1, 0.05])

        from matplotlib.widgets import Button
        self.pause_button = Button(pause_ax, 'Пауза')
        self.pause_button.on_clicked(self.toggle_pause)

        self.stop_button = Button(stop_ax, 'Стоп')
        self.stop_button.on_clicked(self.stop_optimization)

        # Добавление обработчика клавиш
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Добавление легенд
        for ax in self.axes.flat:
            ax.legend()

        # Если есть ручная интерпретация, отобразим её сразу
        if self.manual_interpretation is not None:
            self.plot_manual_interpretation()

        plt.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.001)

    def on_key_press(self, event):
        """Обработчик нажатий клавиш"""
        if event.key == 'p':  # 'p' для паузы
            self.toggle_pause(event)
        elif event.key == 's':  # 's' для остановки
            self.stop_optimization(event)

    def toggle_pause(self, event):
        """Переключить режим паузы"""
        self.paused = not self.paused
        self.pause_button.label.set_text('Продолжить' if self.paused else 'Пауза')
        self.fig.canvas.draw_idle()

        if self.paused:
            # Показываем уведомление о паузе
            plt.figtext(0.5, 0.5, 'ПАУЗА', fontsize=30, ha='center', color='red',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
            self.fig.canvas.draw_idle()

            # Ждем, пока пользователь не продолжит
            while self.paused and not self.should_stop:
                plt.pause(0.1)
                self.fig.canvas.draw_idle()

            # Убираем уведомление о паузе
            self.fig.texts = []
            self.fig.canvas.draw_idle()

    def stop_optimization(self, event):
        """Остановить оптимизацию"""
        self.should_stop = True
        self.paused = False  # Если была пауза, снимаем её

        # Показываем уведомление о остановке
        plt.figtext(0.5, 0.5, 'ОСТАНОВЛЕНО', fontsize=30, ha='center', color='red',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
        self.fig.canvas.draw_idle()

    def plot_manual_interpretation(self):
        """Отображение ручной интерпретации"""
        if self.manual_interpretation is None or self.manual_well is None:
            return

        # Используем индексы из текущих сегментов, чтобы ограничить область отображения
        start_idx = self.segments[0].start_idx
        end_idx = self.segments[-1].end_idx

        # Отображение синтетической кривой от ручной интерпретации
        if np.any(~np.isnan(self.manual_well.synt_curve)):
            self.lines['manual_synth_curve'].set_data(
                self.manual_well.measured_depth[start_idx:end_idx + 1],
                self.manual_well.synt_curve[start_idx:end_idx + 1]
            )

        # Отображение TVT от ручной интерпретации
        if np.any(~np.isnan(self.manual_well.tvt)):
            self.lines['manual_tvt'].set_data(
                self.manual_well.measured_depth[start_idx:end_idx + 1],
                self.manual_well.tvt[start_idx:end_idx + 1]
            )

        # Создаем списки VS и шифтов для отображения
        vs_values = []
        shifts = []

        # Получаем VS из автоматической интерпретации для границ
        segment_start_vs = self.segments[0].start_vs
        segment_end_vs = self.segments[-1].end_vs

        # Получаем шифты из ручной интерпретации для соответствующих индексов
        manual_shift_at_start = get_shift_by_idx(self.manual_interpretation, self.segments[0].start_idx)
        manual_shift_at_end = get_shift_by_idx(self.manual_interpretation, self.segments[-1].end_idx)

        # Добавляем начальную точку с VS из автоматической интерпретации и шифтом из ручной
        vs_values.append(segment_start_vs)
        shifts.append(manual_shift_at_start)

        # Добавляем шифты ручной интерпретации, которые находятся внутри области
        for segment in self.manual_interpretation:
            # Добавляем только те точки, которые строго внутри области интереса
            # (не включая граничные точки)
            if segment.end_vs > segment_start_vs and segment.end_vs < segment_end_vs:
                vs_values.append(segment.end_vs)
                shifts.append(segment.end_shift)

        # Добавляем конечную точку с VS из автоматической интерпретации и шифтом из ручной
        vs_values.append(segment_end_vs)
        shifts.append(manual_shift_at_end)

        # Сортируем точки по VS для правильного отображения
        sorted_points = sorted(zip(vs_values, shifts), key=lambda x: x[0])
        sorted_vs = [point[0] for point in sorted_points]
        sorted_shifts = [point[1] for point in sorted_points]

        # Отображаем шифты ручной интерпретации с границами
        self.lines['manual_shifts'].set_data(sorted_vs, sorted_shifts)

    def update_plot(self, shifts=None, objective_value=None, mse=None, pearson=None):
        """Обновление графиков с текущими данными"""
        if shifts is not None:
            # Обновление истории
            self.history['objective_values'].append(objective_value)
            self.history['shifts'].append(shifts.copy())
            self.history['mse'].append(mse)
            self.history['pearson'].append(pearson)

            # Обновление сегментов
            segments_copy = deepcopy(self.segments)
            for i, segment in enumerate(segments_copy):
                segment.end_shift = shifts[i]
                if i < len(segments_copy) - 1:
                    segments_copy[i + 1].start_shift = shifts[i]

                # Обновляем угол для сегмента
                segment.calc_angle()

            # Удаляем предыдущие аннотации углов
            for ann in self.angle_annotations:
                ann.remove()
            self.angle_annotations = []

            # Расчет траектории
            success = self.well.calc_horizontal_projection(self.typewell, segments_copy)

            if success:
                # Обновление графика кривых
                start_idx = segments_copy[0].start_idx
                end_idx = segments_copy[-1].end_idx
                self.lines['well_curve'].set_data(self.well.measured_depth[start_idx:end_idx + 1],
                                                  self.well.value[start_idx:end_idx + 1])
                self.lines['synth_curve'].set_data(self.well.measured_depth[start_idx:end_idx + 1],
                                                   self.well.synt_curve[start_idx:end_idx + 1])

                # Обновление графика траектории
                self.lines['trajectory'].set_data(self.well.measured_depth[start_idx:end_idx + 1],
                                                  self.well.true_vertical_depth[start_idx:end_idx + 1])
                self.lines['tvt'].set_data(self.well.measured_depth[start_idx:end_idx + 1],
                                           self.well.tvt[start_idx:end_idx + 1])

                # Добавляем отображение углов для каждого сегмента
                for i, segment in enumerate(segments_copy):
                    # Вычисляем координаты середины сегмента для графика траектории
                    x = (self.well.measured_depth[segment.start_idx] + self.well.measured_depth[segment.end_idx]) / 2
                    y = (self.well.tvt[segment.start_idx] + self.well.tvt[segment.end_idx]) / 2

                    # Проверяем, что координаты имеют конечные значения
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue

                    # Создаем смещение ВНИЗ для лучшей видимости
                    y_min = np.nanmin(self.well.tvt[~np.isnan(self.well.tvt)]) if np.any(
                        ~np.isnan(self.well.tvt)) else 0
                    y_max = np.nanmax(self.well.tvt[~np.isnan(self.well.tvt)]) if np.any(
                        ~np.isnan(self.well.tvt)) else 1
                    delta_y = 0.03 * (y_max - y_min)

                    # Отображаем угол НИЖЕ линии сегмента
                    angle_text = self.axes[1, 1].text(x, y - delta_y, f"{segment.angle:.1f}°",
                                                      ha='center', va='top',
                                                      color='blue',
                                                      fontsize=8,
                                                      bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

                    self.angle_annotations.append(angle_text)

            # Обновление графика сдвигов - используем VS вместо индексов
            vs_values = []
            vs_values.append(segments_copy[0].start_vs)
            for segment in segments_copy:
                vs_values.append(segment.end_vs)

            # Создаем список шифтов, включая начальный шифт первого сегмента
            shifts_with_start = [segments_copy[0].start_shift] + list(shifts)

            # Отображаем шифты
            self.lines['shifts'].set_data(vs_values, shifts_with_start)

            # Обновление графика целевой функции
            iterations = np.arange(len(self.history['objective_values']))
            self.lines['objective'].set_data(iterations, self.history['objective_values'])

            # Настройка пределов осей
            for ax in [self.axes[0, 0], self.axes[0, 1], self.axes[1, 0], self.axes[1, 1]]:
                try:
                    ax.relim()
                    ax.autoscale_view()
                except:
                    # Игнорируем ошибки при автомасштабировании
                    pass

            # Обновление итераций
            self.iterations += 1

            # Обновление заголовка с информацией об углах
            # Собираем информацию об углах, только для сегментов с конечными углами
            angle_info = []
            for i, segment in enumerate(segments_copy):
                if hasattr(segment, 'angle') and np.isfinite(segment.angle):
                    angle_info.append(f"Seg{i}: {segment.angle:.1f}°")

            angle_text = ", ".join(angle_info)

            # Обновляем заголовок
            self.fig.suptitle(f'Optimization Progress - Iteration {self.iterations}, '
                              f'Objective: {objective_value:.6f}, MSE: {mse:.6f}\n'
                              f'Angles: {angle_text}', fontsize=12)

            # Обновление отображения
            self.fig.canvas.draw()
            plt.pause(0.1)  # Пауза для обновления отображения

    def close(self):
        """Закрытие визуализации"""
        plt.close(self.fig)
        plt.ioff()


def visualizing_optimizer_fit(well,
                              typewell,
                              self_corr_start_idx,
                              segments,
                              angle_range,
                              angle_sum_power,
                              segm_counts_reg,
                              num_iterations,
                              pearson_power,
                              mse_power,
                              num_intervals_self_correlation,
                              sc_power,
                              min_pearson_value,
                              use_accumulative_bounds=True,
                              multi_threaded=False,
                              manual_interpretation=None,
                              well_manual_interpretation=None):
    """
    Оптимизирует сдвиги сегментов с визуализацией процесса

    Args:
        well: Объект скважины
        typewell: Объект опорной скважины
        self_corr_start_idx: Начальный индекс для корреляции
        segments: Список сегментов для оптимизации
        angle_range: Диапазон углов для ограничений
        segm_counts_reg: Список размеров региональных участков
        num_iterations: Число итераций
        pearson_power, mse_power, num_intervals_self_correlation, sc_power: Параметры для расчета метрики
        min_pearson_value: Минимальное допустимое значение для корреляции Пирсона
        use_accumulative_bounds: Флаг использования накопительного расчета границ
        multi_threaded: Не используется (для совместимости)
        manual_interpretation: Ручная интерпретация для сравнения (опционально)
        well_manual_interpretation: Объект скважины с ручной интерпретацией (опционально)

    Returns:
        Список результатов
    """
    from ag_optimizer_utils import calculate_optimization_bounds, \
        collect_optimization_stats
    from ag_rewards.ag_func_correlations import objective_function_optimizer

    # Создаем визуализатор с передачей всех необходимых параметров
    visualizer = OptimizationVisualizer(
        well=well,
        typewell=typewell,
        segments=segments,
        max_iterations=num_iterations,
        manual_interpretation=manual_interpretation,
        well_manual_interpretation=well_manual_interpretation
    )

    # Задаем ограничения для сдвигов на основе angle_range, используя общую функцию
    bounds = calculate_optimization_bounds(segments, angle_range, use_accumulative_bounds)

    # Начальные сдвиги - текущие конечные сдвиги каждого сегмента
    initial_shifts = [segment.end_shift for segment in segments]

    # Создаем функцию обратного вызова для визуализации в процессе оптимизации
    def callback(xk, convergence=None):
        # Эта функция вызывается на каждой итерации оптимизации
        # Проверяем, нужно ли остановить оптимизацию
        if visualizer.should_stop:
            print("Оптимизация остановлена пользователем")
            return True

        # Вызываем objective_function_optimizer для обновления визуализации
        objective_function_optimizer(
            xk,
            well,
            typewell,
            self_corr_start_idx,
            segments,
            pearson_power,
            mse_power,
            num_intervals_self_correlation,
            sc_power,
            angle_range,
            angle_sum_power,
            min_pearson_value,
            visualizer
        )
        return False  # Продолжаем оптимизацию

    # Используем метод оптимизации из хардкода
    optimization_method = OPTIMIZATION_METHOD

    # Создаем лямбда-функцию для оптимизации
    obj_function = lambda x: objective_function_optimizer(
        x,
        well,
        typewell,
        self_corr_start_idx,
        segments,
        pearson_power,
        mse_power,
        num_intervals_self_correlation,
        sc_power,
        angle_range,
        angle_sum_power,
        min_pearson_value,
        visualizer
    )

    # Выполняем оптимизацию с визуализацией
    if optimization_method == 'differential_evolution':
        result = differential_evolution(
            obj_function,
            bounds=bounds,
            callback=callback,
            popsize=10,
            maxiter=30
        )
    else:
        result = minimize(
            obj_function,
            initial_shifts,
            method=optimization_method,
            bounds=bounds,
            callback=callback
            )

    # Собираем статистику оптимизации
    optimization_stats = collect_optimization_stats(result, well, typewell, segments, pearson_power, mse_power)
    print(f"Optimization stats: {optimization_stats}")

    # Создаем результат с оптимальными сдвигами
    optimal_segments = deepcopy(segments)
    for i, shift in enumerate(result.x):
        optimal_segments[i].end_shift = shift
        if i < len(optimal_segments) - 1:
            optimal_segments[i + 1].start_shift = shift

    # Обновляем TVT в оригинальном объекте well, используя общую функцию
    well.calc_horizontal_projection(typewell, optimal_segments)

    # Рассчитываем итоговую корреляцию для оптимальных сегментов
    well_copy = deepcopy(well)
    well_copy.calc_horizontal_projection(typewell, optimal_segments)
    corr, _, self_correlation, _, _, pearson, num_points, mse, _, _ = calculate_correlation(
        well_copy,
        self_corr_start_idx,
        optimal_segments[0].start_idx,
        optimal_segments[-1].end_idx,
        float('inf'),
        0,
        0,
        pearson_power,
        mse_power,
        num_intervals_self_correlation,
        sc_power,
        min_pearson_value
    )

    # Сообщение о результатах
    if visualizer.should_stop:
        plt.figtext(0.5, 0.5, 'Оптимизация остановлена пользователем',
                    fontsize=20, ha='center', color='red',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
        visualizer.fig.canvas.draw_idle()
        plt.pause(1.0)  # Пауза, чтобы пользователь увидел сообщение

    # Закрываем визуализатор
    visualizer.close()

    # Генерируем результаты с основным оптимальным решением
    results = [(corr, self_correlation, pearson, mse, num_points, optimal_segments, well_copy)]

    # Сохраняем статистику оптимизации в первом результате
    if len(results) > 0 and hasattr(results[0][6], 'optimization_stats'):
        results[0][6].optimization_stats = optimization_stats

    # Если оптимизация не была остановлена пользователем, генерируем дополнительные результаты
    if not visualizer.should_stop:
        # Генерируем дополнительные результаты с небольшими возмущениями вокруг оптимального результата
        for _ in range(min(num_iterations - 1, 9)):
            noisy_segments = deepcopy(optimal_segments)

            # Добавляем небольшие случайные возмущения
            for i, segment in enumerate(noisy_segments):
                segment_len = segment.end_vs - segment.start_vs
                perturbation_size = segment_len * well.horizontal_well_step * np.tan(np.radians(angle_range * 0.1))
                shift_perturbation = random.uniform(-perturbation_size, perturbation_size)

                segment.end_shift += shift_perturbation
                if i < len(noisy_segments) - 1:
                    noisy_segments[i + 1].start_shift = segment.end_shift

            # Рассчитываем корреляцию для возмущенных сегментов
            noisy_well = deepcopy(well)
            success = noisy_well.calc_horizontal_projection(typewell, noisy_segments)
            if success:
                perturbed_corr, _, perturbed_self_correlation, _, _, perturbed_pearson, perturbed_num_points, perturbed_mse, _, _ = calculate_correlation(
                    noisy_well,
                    self_corr_start_idx,
                    noisy_segments[0].start_idx,
                    noisy_segments[-1].end_idx,
                    float('inf'),
                    0,
                    0,
                    pearson_power,
                    mse_power,
                    num_intervals_self_correlation,
                    sc_power,
                    min_pearson_value
                )

                results.append((perturbed_corr, perturbed_self_correlation, perturbed_pearson, perturbed_mse,
                                perturbed_num_points, noisy_segments, noisy_well))

    # Сортируем результаты по корреляции (по убыванию)
    results.sort(key=lambda x: x[0], reverse=True)

    return results


def get_visualizing_optimizer_interpretations_list(
        well
        , type_well
        , num_iterations
        , angle_range
        , angle_sum_power
        , segments_count_curr
        , segment_len_curr
        , segm_counts_reg
        , backstep
        , interpretation_start_idx
        , manl_interpr_start_idx
        , big_segment_start_idx
        , cur_start_shift
        , basic_segments
        , current_all_segments
        , pearson_power
        , mse_power
        , num_intervals_self_correlation
        , sc_power
        , manl_interpr=None
        , well_manl_interp=None
        , min_pearson_value=-1
        , use_accumulative_bounds=True
):
    """
    Визуализирующая версия get_optimizer_interpretations_list
    """

    # ДОБАВЛЯЕМ ИНИЦИАЛИЗАЦИЮ TVT - аналогично алгоритму Монте-Карло:
    # Если есть предыдущие сегменты, то используем их для получения начального смещения
    if current_all_segments:
        # Если это не первая интерпретация, берем смещение из предыдущих сегментов
        cur_start_shift = get_shift_by_idx(current_all_segments, big_segment_start_idx)

    # Инициализируем TVT для начальной точки
    # Это особенно важно для первой итерации или при изменении размера сегментов
    well.tvt[big_segment_start_idx] = well.true_vertical_depth[big_segment_start_idx] + cur_start_shift

    segments = create_segments(
        well,
        segments_count_curr,
        segment_len_curr,
        big_segment_start_idx,
        cur_start_shift,
        basic_segments)

    # Используем визуализирующую версию optimizer_fit с передачей ручной интерпретации
    temp_results = visualizing_optimizer_fit(
        well,
        type_well,
        self_corr_start_idx=min(interpretation_start_idx, manl_interpr_start_idx),
        segments=segments,
        angle_range=angle_range,
        angle_sum_power=angle_sum_power,
        segm_counts_reg=segm_counts_reg,
        num_iterations=num_iterations,
        pearson_power=pearson_power,
        mse_power=mse_power,
        num_intervals_self_correlation=num_intervals_self_correlation,
        sc_power=sc_power,
        min_pearson_value=min_pearson_value,
        use_accumulative_bounds=use_accumulative_bounds,
        multi_threaded=False,
        manual_interpretation=manl_interpr,
        well_manual_interpretation=well_manl_interp
    )

    extended_results = [(current_all_segments + result[5],) + result for result in temp_results]
    return extended_results