import numpy as np
import pandas as pd
from ag_rewards.ag_func_correlations import linear_interpolation
from ag_utils.ag_func_checks import check_uniform


class TypeWell:
    def __init__(self, data_source, typewell_step=0.03048):
        """
        Универсальный конструктор для TypeWell

        Args:
            data_source: dict (JSON typeLog from emulator) или DataFrame (legacy)
            typewell_step: шаг опорной скважины
        """
        self.typewell_step = typewell_step

        if isinstance(data_source, dict):  # JSON from emulator
            self.prepare_data_from_json(data_source)
        else:  # DataFrame (legacy)
            self.prepare_data(data_source)

    def prepare_data_from_json(self, json_data):
        """Подготовка данных из JSON typeLog эмулятора"""
        # Извлекаем typeLog - может быть прямо массивом или в структуре
        if 'typeLog' in json_data:
            type_log_points = json_data['typeLog']
        elif isinstance(json_data, list):  # Прямо массив typeLog
            type_log_points = json_data
        else:
            raise ValueError("No typeLog found in JSON data")

        if not type_log_points:
            raise ValueError("Empty typeLog data")

        # Создаем DataFrame из typeLog
        df_type_well = pd.DataFrame({
            'Depth': [p['trueVerticalDepth'] for p in type_log_points["tvdSortedPoints"]],
            'Curve': [p['data'] for p in type_log_points["tvdSortedPoints"]]
        })

        # Очистка NaN значений
        df_type_well = df_type_well.dropna(subset=['Curve'])

        # Проверка на дубликаты TVD после очистки NaN
        if df_type_well['Depth'].duplicated().any():
            duplicates = df_type_well[df_type_well['Depth'].duplicated(keep=False)]
            raise ValueError(f"Duplicate TVD values found in typeLog after NaN cleanup: {duplicates['Depth'].unique()}")

        # Determine source step from data
        # Use source data directly if uniform and step >= 0.1ft (0.03048m)
        MIN_STEP = 0.03048  # 0.1ft
        depths = df_type_well['Depth'].values
        if len(depths) > 1:
            source_step = np.median(np.diff(depths))
            if source_step >= MIN_STEP and check_uniform(depths, source_step):
                # Source data is uniform with acceptable step - use directly without resample
                self.typewell_step = source_step
                self.min_depth = depths.min()
                self.tvd = depths
                self.value = df_type_well['Curve'].values
                self.normalized = False
                return
            # else: resample to MIN_STEP or source_step

        # Вызываем стандартную подготовку данных (with resample)
        self.prepare_data(df_type_well)

    def prepare_data(self, df_type_well):
        """Стандартная подготовка данных из DataFrame"""
        self.min_depth = df_type_well['Depth'].min()
        max_depth = df_type_well['Depth'].max()
        range_depth = max_depth - self.min_depth

        # Создание новых значений глубины с нужным шагом
        new_depth_values = np.arange(self.min_depth, max_depth + self.typewell_step, self.typewell_step)
        new_df = pd.DataFrame({'Depth': new_depth_values})

        new_df = new_df.set_index('Depth')
        df_type_well = df_type_well.set_index('Depth')

        # Объединение нового и старого DataFrame, установка глубины в качестве индекса для корректной интерполяции
        new_df = pd.concat([new_df, df_type_well], axis=1).sort_index()

        # Интерполяция и сброс индекса для получения обратно колонки Depth
        new_df = new_df.interpolate(method='linear')

        # удаление лишних
        new_df = new_df.loc[new_df.index.isin(new_depth_values)].reset_index()

        self.tvd = new_df['Depth'].values
        self.value = new_df['Curve'].values
        self.normalized = False

        assert check_uniform(self.tvd, self.typewell_step)

    def normalize(self,
                  norm_divider,
                  well_min_depth,
                  md_range):
        self.wells_min_depth = min(self.min_depth, well_min_depth)
        self.tvd = (self.tvd - self.wells_min_depth) / md_range
        self.value = self.value / norm_divider
        self.normalized_min_depth = np.min(self.tvd)
        self.normalized_typewell_step = self.typewell_step / md_range
        self.normalized = True

    def denormalize(self,
                    norm_divider,
                    well_min_depth,
                    md_range):
        self.wells_min_depth = min(self.min_depth, well_min_depth)
        self.tvd = self.tvd * md_range + self.wells_min_depth
        self.value = self.value * norm_divider
        self.normalized = False

    def check_uniform_normalized(self):
        for i, depth in enumerate(self.tvd):
            if i < len(self.tvd - 1):
                next_calculated_depth = depth + self.normalized_typewell_step
                next_depth = self.tvd[i + 1]
                if abs(next_calculated_depth - next_depth) > 1e-15:
                    print('jopa jop typewell')

    def tvt2value(self, tvt):
        # closest_idx_below = np.argmax(self.depth[self.depth <= tvt])
        if self.normalized:
            closest_idx_below = int((tvt - self.normalized_min_depth) / self.normalized_typewell_step)
        else:
            closest_idx_below = int((tvt - self.min_depth) / self.typewell_step)

        # if closest_idx_below != closest_idx_below2:
        #     print('Ne rabotaet poisk bistriy')
        closest_idx_above = closest_idx_below + 1
        depth_below = self.tvd[closest_idx_below]
        depth_above = self.tvd[closest_idx_above]
        curve_below = self.value[closest_idx_below]
        curve_above = self.value[closest_idx_above]
        return linear_interpolation(depth_below, curve_below, depth_above, curve_above, tvt)