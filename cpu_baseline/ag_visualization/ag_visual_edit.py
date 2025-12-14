import matplotlib.pyplot as plt
import numpy as np
from ag_rewards.ag_func_correlations import calculate_correlation
from ag_numerical.ag_func_monte_carlo import update_segments
from ag_objects.ag_obj_well import Well
from ag_objects.ag_obj_typewell import TypeWell
from ag_objects.ag_obj_interpretation import create_segments, normalize_segments
from copy import deepcopy
from ag_utils.ag_func_file_i_o import save_segments_to_csv
from typing import Dict, Any, List, Tuple, Optional, Union


plt.ion()  # Включение интерактивного режима

class InteractivePlot:

    def __init__(
            self,
            well_denorm:Well,
            well: Well,
            segments,
            type_well_denorm:TypeWell,
            type_well: TypeWell,
            well_manual_interpretation_denorm,
            well_manual_interpretation,
            tvd_to_typewell_shift,
            landing_end_md_manual,
            landing_end_md_auto,
            well_calc_params
    ):

        self.well_denorm = well_denorm
        self.well = well
        self.initial_normalized_well = deepcopy(well)
        self.segments_denorm = segments
        self.type_well_denorm = type_well_denorm
        self.type_well = type_well
        self.current_segment_index = 0
        self.well_data_manual_interpretation_denorm = well_manual_interpretation_denorm
        self.well_data_manual_interpretation = well_manual_interpretation
        self.landing_end_md_manual = landing_end_md_manual
        self.landing_end_md_auto = landing_end_md_auto
        self.well_name = well_calc_params['well_name']
        self.well_calc_params = well_calc_params
        self.tvd_to_typewell_shift = tvd_to_typewell_shift

        self.fig = None
        self.axes = []
        self.lines = {}

        self.setup_plot()
        self.draw_initial_data()

        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def setup_plot(self):
        grid = plt.GridSpec(2, 3, width_ratios=[1, 0.5, 0.5], height_ratios=[2, 1], wspace=0.5, hspace=0.5)
        self.fig = plt.figure(figsize=(12, 8))

        ax1 = self.fig.add_subplot(grid[0, 0])
        ax2 = self.fig.add_subplot(grid[1, 0])
        ax3 = self.fig.add_subplot(grid[:, 1])
        ax4 = self.fig.add_subplot(grid[:, 2])
        self.axes = [ax1, ax2, ax3, ax4]

    def draw_initial_data(self):

        self.fig.suptitle(f'{self.well_name[:-5]}', fontsize=14, fontweight='bold')
        ax1, ax2, ax3, ax4 = self.axes

        # ИЗМЕНЕНИЕ 3: Тонкие линии для лучшей читаемости + правильный порядок отрисовки
        self.lines['trajectory'], = ax1.plot([], [], label='Trajectory', color='gray', linewidth=0.8,
                                             alpha=0.7)  # Тонкая серая траектория
        self.lines['manual'], = ax1.plot([], [], label='Manual', color='blue',
                                         linewidth=3.5)  # ТОЛСТАЯ РУЧНАЯ, РИСУЕМ ПЕРВОЙ
        self.lines['geology'], = ax1.plot([], [], label='Auto Geology', color='red', linestyle='-',
                                          linewidth=2.5)  # Автоматическая поверх
        self.lines['current_segment'], = ax1.plot([], [], label='Current Segment', color='orange', linewidth=2.0)
        self.lines['interpolated'], = ax1.plot([], [], label='Interpolated Geology', color='magenta', linewidth=1.0,
                                               alpha=0.7)

        self.lines['well_curve'], = ax2.plot([], [], label='Well', color='red', linewidth=1.0)
        self.lines['synth_curve'], = ax2.plot([], [], label='Auto Synth', color='green', linewidth=1.5)
        self.lines['manual_synth'], = ax2.plot([], [], label='Manual Synth', color='blue',
                                               linewidth=2.0)  # ТОЛСТАЯ РУЧНАЯ

        self.lines['vertical_auto_well_curve'], = ax3.plot([], [], label='Auto horizontal curve', color='red',
                                                           linewidth=1.0)
        self.lines['vertical_typewell_curve'], = ax3.plot([], [], label='Type Well curve', color='black', linewidth=1.0)

        self.lines['vertical_typewell_curve_mn'], = ax4.plot([], [], label='Type Well curve', color='black',
                                                             linewidth=1.0)
        self.lines['vertical_manual_well_curve'], = ax4.plot([], [], label='Manual horizontal curve', color='blue',
                                                             linewidth=1.5)

        self.update_plot_data()

        # Set axis labels and title
        ax1.set_xlabel('MD')
        ax1.set_ylabel('Depth')
        ax1.set_title('Well Trajectory and Geology')

        ax2.set_xlabel('MD')
        ax2.set_ylabel('Gamma')
        ax2.set_title('Horizontal projection')

        ax3.set_xlabel("Curve Value")
        ax3.set_ylabel("Depth")
        ax3.set_title("Auto: Projection of Curve onto Reference Well")

        ax4.set_xlabel("Curve Value")
        ax4.set_ylabel("Depth")
        ax4.set_title("Manual: Projection of Curve onto Reference Well")

        ax3.set_ylim(ax3.get_ylim()[::-1])
        ax4.set_ylim(ax4.get_ylim()[::-1])

        for ax in self.axes:
            ax.legend()
            ax.grid(True, alpha=0.3)  # Добавляем легкую сетку для лучшей читаемости

    def update_plot_data(self):
        # Константы для улучшения визуализации
        VERTICAL_SCALE_FACTOR = 3.0  # Увеличиваем вертикальный масштаб в 3 раза

        # Обновляем проекцию только если есть сегменты
        if self.segments_denorm:
            self.well_denorm.calc_horizontal_projection(self.type_well_denorm,
                                                        self.segments_denorm,
                                                        self.tvd_to_typewell_shift)

            # ИСПРАВЛЕНИЕ: используем уже нормализованные сегменты из update_interpretation
            if hasattr(self, 'segments') and self.segments:
                save_segments_to_csv(self.segments, '../segment_2.csv')
                self.well.calc_horizontal_projection(self.type_well,
                                                     self.segments,
                                                     self.tvd_to_typewell_shift / self.well.md_range)
            else:
                # Fallback: создаем нормализованные сегменты если их нет
                from ag_objects.ag_obj_interpretation import normalize_segments
                self.segments = normalize_segments(self.segments_denorm, self.well_denorm.md_range)
                save_segments_to_csv(self.segments, '../segment_2.csv')
                self.well.calc_horizontal_projection(self.type_well,
                                                     self.segments,
                                                     self.tvd_to_typewell_shift / self.well.md_range)

        # Корреляция только если есть И manual И auto интерпретация
        if (self.segments_denorm and
                self.well_data_manual_interpretation_denorm and
                len(self.segments_denorm) > 0):

            corr_auto, \
                best_corr_auto, \
                self_correlation_auto, \
                mean_self_correlation_intersect_auto, \
                num_try_with_intersection_auto, \
                pearson_auto, \
                num_points_auto, \
                mse_auto, \
                intersections_mult_auto, \
                intersections_count_auto = calculate_correlation(
                self.well,
                self.segments[0].start_idx,
                start_idx=self.segments[0].start_idx,
                end_idx=self.segments[self.current_segment_index].end_idx,
                mean_self_correlation_intersect=1,
                num_try_with_intersection=1,
                best_corr=1,
                pearson_power=self.well_calc_params['pearson_power'],
                mse_power=self.well_calc_params['mse_power'],
                num_intervals_self_correlation=self.well_calc_params['num_intervals_sc'],
                sc_power=self.well_calc_params['sc_power'],
                min_pearson_value=self.well_calc_params['min_pearson_value']
            )

            corr_manual, \
                best_corr_manual, \
                self_correlation_manual, \
                mean_self_correlation_intersect_manual, \
                num_try_with_intersection_manual, \
                pearson_manual, \
                num_points_manual, \
                mse_manual, \
                intersections_mult_manual, \
                intersections_count_manual = calculate_correlation(
                self.well_data_manual_interpretation,
                self.segments_denorm[0].start_idx,
                start_idx=self.segments_denorm[0].start_idx,
                end_idx=self.segments_denorm[self.current_segment_index].end_idx,
                mean_self_correlation_intersect=1,
                num_try_with_intersection=1,
                best_corr=1,
                pearson_power=self.well_calc_params['pearson_power'],
                mse_power=self.well_calc_params['mse_power'],
                num_intervals_self_correlation=self.well_calc_params['num_intervals_sc'],
                sc_power=self.well_calc_params['sc_power'],
                min_pearson_value=self.well_calc_params['min_pearson_value']
            )
        else:
            # Если нет данных для корреляции - устанавливаем значения по умолчанию
            corr_auto = 0.0
            pearson_auto = 0.0
            intersections_count_auto = 0
            corr_manual = 0.0
            intersections_count_manual = 0

        ax1 = self.axes[0]
        ax3 = self.axes[2]
        ax4 = self.axes[3]

        if hasattr(self, 'corr_auto_text'):
            self.corr_auto_text.remove()
        if hasattr(self, 'corr_manual_text'):
            self.corr_manual_text.remove()
        if hasattr(self, 'angle_text'):
            self.angle_text.remove()

        self.corr_auto_text = ax3.text(0.05, -0.1,
                                       f'Auto Corr: {corr_auto:.8f}, Inters count: {intersections_count_auto:.3f}',
                                       transform=ax3.transAxes,
                                       fontsize=10,
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        self.corr_manual_text = ax4.text(0.05, -0.1,
                                         f'Manual Corr: {corr_manual:.3f}, Inters count: {intersections_count_manual:.3f}',
                                         transform=ax4.transAxes,
                                         fontsize=10,
                                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Угол только если есть сегменты
        if self.segments_denorm:
            angle_auto = self.segments[self.current_segment_index].angle if len(
                self.segments) > self.current_segment_index else 0
            angle_denorm = self.segments_denorm[self.current_segment_index].angle if len(
                self.segments_denorm) > self.current_segment_index else 0
        else:
            angle_auto = 0
            angle_denorm = 0

        self.angle_text = ax1.text(0.05, -0.25, f'Angle: {angle_auto:.4f},'
                                                f'Angle: {angle_denorm:.4f},'
                                                f' Pearson auto: {pearson_auto:.10f}',
                                   transform=ax1.transAxes,
                                   fontsize=10,
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # ИЗМЕНЕНИЕ 1: Показываем всю траекторию скважины целиком
        self.lines['trajectory'].set_data(
            self.well_denorm.measured_depth,  # ВСЯ СКВАЖИНА
            -self.well_denorm.true_vertical_depth  # ВСЯ СКВАЖИНА
        )

        # Остальная логика обновления графиков (только если есть сегменты)
        if self.segments_denorm:
            # Получение индексов начала и конца текущего сегмента
            start_md = self.well_denorm.measured_depth[self.segments_denorm[self.current_segment_index].start_idx]
            end_md = self.well_denorm.measured_depth[self.segments_denorm[self.current_segment_index].end_idx]

            # Добавление вертикальных линий на geometry_graph и horizont_curve
            geometry_graph, horizont_curve = self.axes[0], self.axes[1]

            # Удаление предыдущих линий, если они существуют
            if hasattr(self, 'start_line_ax1'):
                self.start_line_ax1.remove()
                self.end_line_ax1.remove()
            if hasattr(self, 'start_line_ax2'):
                self.start_line_ax2.remove()
                self.end_line_ax2.remove()

            # Добавление новых линий
            self.start_line_ax1 = geometry_graph.axvline(x=start_md, color='magenta', linestyle='--')
            self.end_line_ax1 = geometry_graph.axvline(x=end_md, color='cyan', linestyle='--')

            self.start_line_ax2 = horizont_curve.axvline(x=start_md, color='magenta', linestyle='--')
            self.end_line_ax2 = horizont_curve.axvline(x=end_md, color='cyan', linestyle='--')

            self.landing_end_manual_line_ax1 = geometry_graph.axvline(x=self.landing_end_md_manual, color='k',
                                                                      linestyle='--')
            self.landing_end_manual_line_ax2 = geometry_graph.axvline(x=self.landing_end_md_manual, color='k',
                                                                      linestyle='--')

            self.landing_end_auto_line_ax1 = geometry_graph.axvline(x=self.landing_end_md_auto, color='r',
                                                                    linestyle='--')
            self.landing_end_auto_line_ax2 = geometry_graph.axvline(x=self.landing_end_md_auto, color='r',
                                                                    linestyle='--')

            interpolated_shifts = np.zeros_like(self.well_denorm.measured_depth)
            for segment in self.segments_denorm:
                interpolated_shifts[segment.start_idx:segment.end_idx + 1] = np.linspace(
                    segment.start_shift, segment.end_shift, segment.end_idx - segment.start_idx + 1)

            # Автоматическая геология - только в диапазоне сегментов (дорисовывается)
            self.lines['geology'].set_data(self.well_denorm.measured_depth[
                                           self.segments_denorm[0].start_idx: self.segments_denorm[-1].end_idx + 1],
                                           - self.well_denorm.true_vertical_depth[
                                             self.segments_denorm[0].start_idx: self.segments_denorm[-1].end_idx + 1]
                                           + self.well_denorm.tvt[
                                             self.segments_denorm[0].start_idx: self.segments_denorm[-1].end_idx + 1]
                                           - self.well_denorm.tvt[self.segments_denorm[0].start_idx])

            self.lines['current_segment'].set_data(self.well_denorm.measured_depth[
                                                   self.segments_denorm[self.current_segment_index].start_idx:
                                                   self.segments_denorm[self.current_segment_index].end_idx + 1],
                                                   - self.well_denorm.true_vertical_depth[
                                                     self.segments_denorm[self.current_segment_index].start_idx:
                                                     self.segments_denorm[self.current_segment_index].end_idx + 1]
                                                   + self.well_denorm.tvt[
                                                     self.segments_denorm[self.current_segment_index].start_idx:
                                                     self.segments_denorm[self.current_segment_index].end_idx + 1]
                                                   - self.well_denorm.tvt[
                                                       self.segments_denorm[self.current_segment_index].start_idx])

            self.lines['interpolated'].set_data(
                self.well_denorm.measured_depth[self.segments[0].start_idx: self.segments[-1].end_idx + 1],
                self.well_denorm.tvt[self.segments[0].start_idx] -
                interpolated_shifts[self.segments[0].start_idx: self.segments[-1].end_idx + 1]
                )

            # Обновление линий horizont_curve Horizontal projection
            self.lines['well_curve'].set_data(
                self.well_denorm.measured_depth[self.segments_denorm[0].start_idx:self.segments_denorm[-1].end_idx + 1],
                self.well_denorm.value[
                self.segments_denorm[0].start_idx:self.segments_denorm[-1].end_idx + 1])

            self.lines['synth_curve'].set_data(
                self.well_denorm.measured_depth[self.segments_denorm[0].start_idx:self.segments_denorm[-1].end_idx + 1],
                self.well_denorm.synt_curve[
                self.segments_denorm[0].start_idx:self.segments_denorm[-1].end_idx + 1])

            # Обновление линий vert_curve_aut Projection of Curve onto the Reference Well
            self.lines['vertical_auto_well_curve'].set_data(self.well_denorm.value, -self.well_denorm.tvt)
            self.lines['vertical_typewell_curve'].set_data(self.type_well_denorm.value, -self.type_well_denorm.tvd)

            start_idx = self.segments_denorm[self.current_segment_index].start_idx
            end_idx = self.segments_denorm[self.current_segment_index].end_idx

            # Удаление зеленой кривой, если она уже существует
            if hasattr(self, 'green_curve_segment'):
                self.green_curve_segment.remove()

            # Добавление зеленой кривой для текущего сегмента
            vert_curve_aut = self.axes[2]
            self.green_curve_segment, = vert_curve_aut.plot(
                self.well_denorm.value[start_idx:end_idx + 1],
                self.well_denorm.tvt[start_idx:end_idx + 1],
                color='green'
            )

        # Обновляем ручную интерпретацию если есть (показываем всю)
        if self.well_data_manual_interpretation_denorm:
            # ИСПРАВЛЕНИЕ: Безопасное отображение всей ручной интерпретации
            manual_tvt = self.well_data_manual_interpretation_denorm.tvt
            manual_depth = self.well_data_manual_interpretation_denorm.true_vertical_depth

            # Находим первую валидную точку для нормализации
            valid_indices = ~np.isnan(manual_tvt)
            if np.any(valid_indices):
                first_valid_tvt = manual_tvt[valid_indices][0]

                self.lines['manual'].set_data(
                    self.well_data_manual_interpretation_denorm.measured_depth,  # ВСЯ РУЧНАЯ ИНТЕРПРЕТАЦИЯ
                    -manual_depth + manual_tvt - first_valid_tvt  # Нормализуем относительно первой валидной точки
                )

                # Горизонтальная проекция - вся ручная интерпретация
                self.lines['manual_synth'].set_data(
                    self.well_data_manual_interpretation_denorm.measured_depth,  # ВСЯ РУЧНАЯ ИНТЕРПРЕТАЦИЯ
                    self.well_data_manual_interpretation_denorm.synt_curve
                )
            else:
                # Если все NaN - показываем пустые линии
                self.lines['manual'].set_data([], [])
                self.lines['manual_synth'].set_data([], [])

            self.lines['vertical_manual_well_curve'].set_data(self.well_data_manual_interpretation_denorm.value,
                                                              -self.well_data_manual_interpretation_denorm.tvt)
            self.lines['vertical_typewell_curve_mn'].set_data(self.type_well_denorm.value, -self.type_well_denorm.tvd)

        self.fig.canvas.draw_idle()

        # ИЗМЕНЕНИЕ 2: Установка пределов графиков с фокусом на геологию
        geometry_graph, horizont_curve, vert_curve_aut, vert_curve_man = self.axes

        # Определяем X диапазон: от начала ручной интерпретации
        if self.well_data_manual_interpretation_denorm:
            # Начинаем с первой точки ручной интерпретации
            min_md = self.well_data_manual_interpretation_denorm.measured_depth[0]
        else:
            # Fallback: начинаем с начала скважины
            min_md = min(self.well_denorm.measured_depth)

        max_md = max(self.well_denorm.measured_depth)

        # Определяем Y диапазон: от нижней точки траектории до верхней точки геологии
        y_coords = []

        # Нижняя граница: самая глубокая точка траектории
        trajectory_y_min = np.min(-self.well_denorm.true_vertical_depth)
        y_coords.append(trajectory_y_min)

        # Верхняя граница: самая высокая точка геологии
        if self.well_data_manual_interpretation_denorm:
            # Ручная интерпретация
            manual_tvt = self.well_data_manual_interpretation_denorm.tvt
            manual_depth = self.well_data_manual_interpretation_denorm.true_vertical_depth

            valid_indices = ~np.isnan(manual_tvt)
            if np.any(valid_indices):
                first_valid_tvt = manual_tvt[valid_indices][0]
                manual_geology_y = -manual_depth + manual_tvt - first_valid_tvt
                y_coords.extend(manual_geology_y[valid_indices])

        # Автоматическая геология (если есть)
        if self.segments_denorm:
            _, geology_y = self.lines['geology'].get_data()
            if len(geology_y) > 0:
                y_coords.extend(geology_y)

        if len(y_coords) > 1:
            y_min, y_max = min(y_coords), max(y_coords)

            # Добавляем небольшой буфер (5% от диапазона)
            y_range = y_max - y_min
            y_buffer = y_range * 0.05

            # Установка пределов для geometry_graph с фокусом на геологию
            geometry_graph.set_xlim(min_md, max_md)
            geometry_graph.set_ylim(y_min - y_buffer, y_max + y_buffer)

            horizont_curve.set_xlim(min_md, max_md)
            horizont_curve.set_ylim(min(self.well_denorm.value), max(self.well_denorm.value))

            tvt_valid = ~np.isnan(self.well_denorm.tvt)

            if np.any(tvt_valid):
                min_tvt = np.nanmin(-self.well_denorm.tvt[tvt_valid])
                max_tvt = np.nanmax(-self.well_denorm.tvt[tvt_valid])
            else:
                # Use reasonable defaults if no auto interpretation yet
                min_tvt = -5000  # Default depth range
                max_tvt = 0

            if self.well_data_manual_interpretation_denorm:
                manual_min_tvt = np.nanmin([-self.well_data_manual_interpretation_denorm.tvt])
                manual_max_tvt = np.nanmax([-self.well_data_manual_interpretation_denorm.tvt])

                # Безопасное обновление min/max с проверкой на NaN
                if not np.isnan(manual_min_tvt):
                    if np.isnan(min_tvt):
                        min_tvt = manual_min_tvt
                    else:
                        min_tvt = min(min_tvt, manual_min_tvt)

                if not np.isnan(manual_max_tvt):
                    if np.isnan(max_tvt):
                        max_tvt = manual_max_tvt
                    else:
                        max_tvt = max(max_tvt, manual_max_tvt)

            vert_curve_aut.set_xlim(min(self.well_denorm.value), max(self.well_denorm.value))
            vert_curve_aut.set_ylim(min_tvt, max_tvt)

            if self.well_data_manual_interpretation_denorm:
                vert_curve_man.set_xlim(min(self.well_data_manual_interpretation_denorm.value),
                                        max(self.well_data_manual_interpretation_denorm.value))
                vert_curve_man.set_ylim(min_tvt, max_tvt)
        else:
            # Fallback: используем стандартное масштабирование
            geometry_graph.set_xlim(min_md, max_md)
            geometry_graph.set_ylim(-1000, 100)  # Разумные значения по умолчанию

        self.fig.canvas.draw_idle()

    def update_interpretation(self, interpretation_data: Dict[str, Any], current_md: float):
        """Обновление интерпретации из эмулятора"""
        if not interpretation_data or 'interpretation' not in interpretation_data:
            return

        # Конвертируем JSON сегменты в AG объекты
        json_segments = interpretation_data['interpretation'].get('segments', [])
        if not json_segments:
            return

        # Создаем новые сегменты из JSON с current_md для правильного endMd
        from ag_objects.ag_obj_interpretation import create_segments_from_json, normalize_segments

        # Создаем денормализованные сегменты
        self.segments_denorm = create_segments_from_json(json_segments, self.well_denorm, current_md)

        # ИСПРАВЛЕНИЕ: Создаем нормализованные сегменты для работы с нормализованной скважиной
        self.segments = normalize_segments(self.segments_denorm, self.well_denorm.md_range)

        # ИЗМЕНЕНИЕ: Используем новый метод обновления
        self.update_visualization()

        print(f"Visualization updated with {len(self.segments_denorm)} segments at MD={current_md:.1f}")

    def shift_point(self, shift):
        step_multiplier = 1 / self.well_denorm.md_range if self.well_denorm.normalized else 1
        self.segments_denorm[self.current_segment_index].end_shift += shift * step_multiplier
        if self.current_segment_index != len(self.segments_denorm) - 1:
            self.segments_denorm[self.current_segment_index + 1].start_shift += shift * step_multiplier

        self.segments_denorm[self.current_segment_index].calc_angle()
        self.segments[self.current_segment_index].calc_angle()



    def on_key(self, event):
        if event.key == 'left':
            self.current_segment_index = max(0, self.current_segment_index - 1)
        elif event.key == 'right':
            self.current_segment_index = min(len(self.segments_denorm) - 1, self.current_segment_index + 1)
        elif event.key == 'up':
            self.shift_point(0.1)
        elif event.key == 'down':
            self.shift_point(-0.1)
        elif event.key == 'pageup':
            self.shift_point(1)
        elif event.key == 'pagedown':
            self.shift_point(-1)
        elif event.key == 'r':
            self.segments_denorm = create_segments(self.well_denorm,
                                                   len(self.segments_denorm),
                                                   self.segments_denorm[1].end_idx - self.segments_denorm[1].start_idx,
                                                   self.segments_denorm[1].start_idx,
                                                   self.segments_denorm[1].start_shift)

            self.segments_denorm = update_segments(self.segments_denorm,
                                                   well=self.well_denorm) # эти параметры прокинуть сверху
        else:
            return

        self.update_plot_data()

    def show(self, block=False, save_to_file=False, file_name='output_image.png'):
        """Оригинальный метод - теперь только для финального показа"""
        if save_to_file:
            self.fig.set_size_inches(24, 16)
            plt.savefig(file_name, dpi=300)
        elif block:
            plt.ioff()  # Выключаем интерактивный режим для блокирующего показа
            plt.show(block=True)
        else:
            # Для совместимости - делаем то же что show_initial
            self.show_initial()

    def show_initial(self):
        """Показать график изначально в неблокирующем режиме на правой половине экрана"""
        # Get screen dimensions
        import tkinter as tk
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()

        # Right half positioning
        window_width = screen_width // 2 - 100  # Half width minus margins
        window_height = int(screen_height * 0.7)  # 70% of screen height
        x_position = screen_width // 2 + 50  # Right half + margin

        plt.ion()  # Интерактивный режим
        plt.show(block=False)  # Неблокирующий показ

        # Position window on right side - handle different backends
        manager = plt.get_current_fig_manager()
        backend = plt.get_backend().lower()

        if 'qt' in backend:
            # Qt backend
            manager.window.setGeometry(x_position, 50, window_width, window_height)
        elif 'tk' in backend:
            # Tkinter backend
            manager.window.wm_geometry(f"{window_width}x{window_height}+{x_position}+50")
        else:
            # Try Qt method as fallback
            manager.window.setGeometry(x_position, 50, window_width, window_height)

        plt.pause(0.1)
        print("AG visualization window opened on right side")

    def update_visualization(self):
        """Обновить визуализацию без повторного показа"""
        try:
            # Обновляем данные графика
            self.update_plot_data()

            # Принудительное обновление canvas
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)  # Небольшая пауза для обновления GUI

        except Exception as e:
            print(f"Failed to update visualization: {e}")

    def close(self):
        """Close visualization window"""
        plt.close(self.fig)