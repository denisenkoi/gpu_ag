import numpy as np
import pandas as pd
from ag_rewards.ag_func_correlations import linear_interpolation
import math
from ag_utils.ag_func_checks import check_uniform
from ag_objects.ag_obj_typewell import TypeWell
import csv
from pathlib import Path

class Well:
    def __init__(self, data_source):
        """
        Универсальный конструктор для Well

        Args:
            data_source: dict (JSON from emulator) или DataFrame (legacy)
        """
        self.normalized = False

        if isinstance(data_source, dict):  # JSON from emulator
            self.prepare_data_from_json(data_source)
        else:  # DataFrame (legacy)
            self.prepare_data(data_source)

    def prepare_data_from_json(self, json_data):
        """Подготовка данных из JSON эмулятора с корректным объединением на общий грид"""
        # Extract trajectory data
        well_points = json_data['well']['points']
        well_log_points = json_data['wellLog']['points']

        assert well_points, "Missing well trajectory data in JSON"
        assert well_log_points, "Missing well log data in JSON"

        # Extract trajectory arrays
        trajectory_md = np.array([p['measuredDepth'] for p in well_points])
        trajectory_tvd = np.array([p['trueVerticalDepth'] for p in well_points])
        trajectory_ns = np.array([p['northSouth'] for p in well_points])
        trajectory_ew = np.array([p['eastWest'] for p in well_points])

        # Extract log arrays
        log_md = np.array([p['measuredDepth'] for p in well_log_points])
        log_curve = np.array([p['data'] for p in well_log_points])

        # Remove NaN values from log data
        valid_log_mask = np.array([x is not None for x in log_curve])
        log_curve = log_curve[valid_log_mask]
        log_md = log_md[valid_log_mask]

        # Convert to float and then check for NaN
        log_curve = log_curve.astype(float)
        additional_nan_mask = ~np.isnan(log_curve)
        log_curve = log_curve[additional_nan_mask]
        log_md = log_md[additional_nan_mask]

        # Check for duplicate MD values
        assert not pd.Series(trajectory_md).duplicated().any(), \
            f"Duplicate MD values in trajectory: {pd.Series(trajectory_md)[pd.Series(trajectory_md).duplicated()]}"

        if pd.Series(log_md).duplicated().any():
            pass

        assert not pd.Series(log_md).duplicated().any(), \
            f"Duplicate MD values in log: {pd.Series(log_md)[pd.Series(log_md).duplicated()]}"

        # Create common MD grid
        self.horizontal_well_step = 0.3048
        min_md = min(trajectory_md.min(), log_md.min())
        max_md = max(trajectory_md.max(), log_md.max())
        common_md = np.arange(min_md, max_md + self.horizontal_well_step, self.horizontal_well_step)

        # Interpolate trajectory data to common grid
        interp_tvd = np.interp(common_md, trajectory_md, trajectory_tvd)
        interp_ns = np.interp(common_md, trajectory_md, trajectory_ns)
        interp_ew = np.interp(common_md, trajectory_md, trajectory_ew)

        # Calculate VS_THL from interpolated coordinates
        dx = np.diff(interp_ns, prepend=interp_ns[0])
        dy = np.diff(interp_ew, prepend=interp_ew[0])
        step_distances = np.sqrt(dx ** 2 + dy ** 2)
        step_distances[0] = 0
        interp_vs = np.cumsum(step_distances)

        # Interpolate log data to common grid (only within log range, NaN outside)
        interp_curve = np.full(len(common_md), np.nan)

        # Find indices within log data range
        log_range_mask = (common_md >= log_md.min()) & (common_md <= log_md.max())

        # Interpolate only within log range
        interp_curve[log_range_mask] = np.interp(
            common_md[log_range_mask],
            log_md,
            log_curve
        )

        # Create final DataFrame with all data on common grid
        df_combined = pd.DataFrame({
            'MD': common_md,
            'Depth': interp_tvd,
            'NorthSouth': interp_ns,
            'EastWest': interp_ew,
            'VS': interp_vs,
            'Curve': interp_curve
        })

        # Call standard data preparation
        self.prepare_data(df_combined)

    def prepare_data(self, df_well):
        """Стандартная подготовка данных из DataFrame"""
        self.horizontal_well_step = 0.3048

        self.min_md = df_well['MD'].min()
        self.max_md = df_well['MD'].max()

        # Create new regular MD grid
        new_md_values = np.arange(self.min_md, self.max_md + self.horizontal_well_step, self.horizontal_well_step)
        new_df = pd.DataFrame({'MD': new_md_values})

        # Interpolate each field separately from original to new grid
        original_md = df_well['MD'].values

        for column in df_well.columns:
            if column != 'MD':
                # Interpolate each field individually to preserve monotonicity
                new_df[column] = np.interp(new_md_values, original_md, df_well[column].values)

        # Финальная очистка NaN после интерполяции и пересчет min и max md
        new_df = new_df.dropna(subset=['Curve'])
        self.min_md = new_df['MD'].min()
        self.max_md = new_df['MD'].max()

        # Set data arrays
        self.measured_depth = new_df['MD'].values
        self.vs_thl = new_df['VS'].values
        self.true_vertical_depth = new_df['Depth'].values
        self.value = new_df['Curve'].values
        self.min_curve, self.max_curve = new_df['Curve'].values.min(), new_df['Curve'].values.max()
        self.tvt = np.empty(len(self.measured_depth))
        self.tvt[:] = np.nan
        self.synt_curve = np.empty(len(self.measured_depth))
        self.synt_curve[:] = np.nan

        self.min_vs = self.vs_thl.min()
        self.min_depth = self.true_vertical_depth.min()
        self.md_range = self.max_md - self.min_md

        # Verify uniformity of MD grid
        assert check_uniform(self.measured_depth, self.horizontal_well_step)

    def check_uniform_normalized(self):
        for i, depth in enumerate(self.measured_depth):
            if i < len(self.vs_thl - 1):
                next_calculated_md = depth + self.normalized_md_step
                next_md = self.true_vertical_depth[i + 1]
                if abs(next_calculated_md - next_md) > 1e-15:
                    print('jopa jop horizontal well normalized')

    def normalize(self,
                  max_curve_value,
                  min_typewell_depth,
                  fixed_md_range=None):
        """
        Normalize well geometry and curve values.

        Args:
            max_curve_value: Maximum curve value for normalization
            min_typewell_depth: Minimum typewell depth for TVD offset
            fixed_md_range: Optional fixed MD range for planning horizon normalization.
                           If None, uses self.md_range (dynamic, changes with well growth).
                           If provided, uses fixed value for consistent normalization across iterations.
        """
        self.max_curve_value = max_curve_value
        self.normalized = True
        # Use fixed_md_range if provided, otherwise use dynamic md_range
        self.normalization_md_range = fixed_md_range if fixed_md_range is not None else self.md_range
        self.horizontal_well_step_norm = self.horizontal_well_step / self.normalization_md_range
        self.wells_min_depth = min(min_typewell_depth, self.min_depth)
        self.value = self.value / max_curve_value
        self.measured_depth = (self.measured_depth - self.min_md) / self.normalization_md_range
        self.vs_thl = (self.vs_thl - self.min_vs) / self.normalization_md_range
        self.true_vertical_depth = (self.true_vertical_depth - self.wells_min_depth) / self.normalization_md_range
        self.min_curve, self.max_curve = self.value.min(), self.value.max()
        assert check_uniform(self.measured_depth, self.horizontal_well_step_norm)

    def denormalize(self):
        self.normalized = False
        self.value = self.value * self.max_curve_value
        # Use normalization_md_range that was set during normalize()
        md_range = getattr(self, 'normalization_md_range', self.md_range)
        self.measured_depth = self.measured_depth * md_range + self.min_md
        self.vs_thl = self.vs_thl * md_range + self.min_vs
        self.true_vertical_depth = self.true_vertical_depth * md_range + self.wells_min_depth

    def calc_synt_curve(self,
                        typewell: TypeWell,
                        tvt_values,
                        synt_curve_values):

        for i, curr_depth in enumerate(tvt_values):
            if np.isnan(curr_depth) or curr_depth < typewell.tvd[0] or curr_depth > typewell.tvd[-1]:
                continue

            synt_curve_values[i] = typewell.tvt2value(tvt_values[i])

        return synt_curve_values

    def calc_segment_tvt(self,
                         typewell,
                         segment,
                         tvd_to_typewell_shift):
        if segment.start_shift is None or segment.end_shift is None:
            print('segment.end_shift is None')
        depth_shift = segment.end_shift - segment.start_shift
        segment_indices = np.array(range(segment.start_idx, segment.end_idx + 1))

        shifts = (segment.start_shift + depth_shift * (self.vs_thl[segment_indices] - segment.start_vs) / (
                    self.vs_thl[segment.end_idx] - self.vs_thl[segment.start_idx]))

        if (self.vs_thl[segment.end_idx] - self.vs_thl[segment.start_idx]) == 0:
            print('delta VS iz ZERO when calc_segment_tvt')
        self.tvt[segment_indices] = ( self.true_vertical_depth[segment_indices] - shifts - tvd_to_typewell_shift)

        if np.isnan(self.tvt[segment_indices]).any():
            return False
        return True

    def calc_horizontal_projection(self,
                                   typewell,
                                   segments,
                                   tvd_to_typewell_shift,
                                   segments_nums=None):
        if segments_nums == None:
            segments_list = segments
        else:
            segments_list = [segments[segments_num] for segments_num in segments_nums]

        # Проверка на пустые сегменты - возвращаем успех, ничего не рисуем
        if not segments_list:
            return True

        for segment in segments_list:
            success = self.calc_segment_tvt(typewell, segment, tvd_to_typewell_shift)
            if not success:
                return False

        segments_indices = np.array(range(segments_list[0].start_idx, segments_list[-1].end_idx + 1))
        # Strict check: ALL points must be within typewell bounds (no +0.3 tolerance)
        # This prevents "cheating" by moving segments outside typewell range
        tvt_min = np.min(typewell.tvd)
        tvt_max = np.max(typewell.tvd)
        if np.any(self.tvt[segments_indices] > tvt_max) or \
                np.any(self.tvt[segments_indices] < tvt_min):
            return False

        # start_time = time.time()
        self.synt_curve[segments_indices] = self.calc_synt_curve(typewell,
                                                                 self.tvt[segments_indices],
                                                                 self.synt_curve[segments_indices])

        return True

    def first_valid_tvt_index(self):
        valid_indices = np.where(~np.isnan(self.tvt))[0]
        if valid_indices.size == 0:
            return None
        return valid_indices[0]

    def trim_data(self,
                  first_idx,
                  max_idx):
        self.measured_depth = self.measured_depth[first_idx: max_idx + 1]
        self.vs_thl = self.vs_thl[first_idx: max_idx + 1]
        self.true_vertical_depth = self.true_vertical_depth[first_idx: max_idx + 1]
        self.value = self.value[first_idx: max_idx + 1]
        self.tvt = self.tvt[first_idx: max_idx + 1]
        self.synt_curve = self.synt_curve[first_idx: max_idx + 1]

    def md2idx(
            self,
            md,
            type='nearest'):

        if type == 'nearest':
            differences = np.abs(self.measured_depth - md)
            closest_index = np.argmin(differences)
            return closest_index
        elif type == 'left':
            return len(self.measured_depth[self.measured_depth < md]) - 1

    def compare_wells(self, well2):
        differences = []
        attributes = [attr for attr in dir(self) if not attr.startswith('__') and not callable(getattr(self, attr))]

        for attr in attributes:
            attr1 = getattr(self, attr)
            attr2 = getattr(well2, attr)

            if isinstance(attr1, np.ndarray) and isinstance(attr2, np.ndarray):
                if not np.array_equal(attr1, attr2):
                    differences.append(f"Атрибут {attr} различается. Массивы не идентичны.")
            else:
                if attr1 != attr2:
                    differences.append(f"Атрибут {attr} различается: {attr1} != {attr2}")

        return differences

    def md2vs(self,
              md,
              well):
        left_idx = self.md2idx(md, 'left')
        l_vs = self.vs_thl[left_idx]
        r_vs = self.vs_thl[left_idx + 1]
        l_md = self.measured_depth[left_idx]
        r_md = self.measured_depth[left_idx + 1]
        return linear_interpolation(l_md, l_vs, r_md, r_vs, md)

    def get_landing_end_md(self, interpretation_start):
        angle_th = 0
        for i in range(interpretation_start, len(self.measured_depth) - 2):
            delta_tvt = self.tvt[i + 1] - self.tvt[i]
            delta_vs = self.vs_thl[i + 1] - self.vs_thl[i]
            angle_well_geology = math.degrees(math.atan(delta_tvt / delta_vs))
            if angle_well_geology <= angle_th:
                return self.measured_depth[i], i
        return None, None

    def export_to_csv(self, csv_path: str, use_feet: bool = True):
        """
        Export well data to CSV file

        Args:
            csv_path: Path to output CSV file
            use_feet: If True, convert meters to feet (divide by 0.3048)
        """
        conversion_factor = 1.0 / 0.3048 if use_feet else 1.0

        # Create output directory if it doesn't exist
        output_path = Path(csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write header
            units_suffix = "_ft" if use_feet else "_m"
            writer.writerow([
                f'MD{units_suffix}',
                f'TVT{units_suffix}',
                f'INC_deg',
                f'TVD{units_suffix}',
                f'VS{units_suffix}',
                'CURVE_value',
                'SYNT_CURVE_value'
            ])

            # Write data rows
            for i in range(len(self.measured_depth)):
                # Convert to feet if requested
                md = self.measured_depth[i] * conversion_factor
                tvt = self.tvt[i] * conversion_factor if not np.isnan(self.tvt[i]) else ''
                tvd = self.true_vertical_depth[i] * conversion_factor
                vs = self.vs_thl[i] * conversion_factor

                # Calculate inclination in degrees from trajectory
                # For well data, we need to calculate inclination from TVD changes
                if i < len(self.measured_depth) - 1:
                    delta_md = self.measured_depth[i + 1] - self.measured_depth[i]
                    delta_tvd = self.true_vertical_depth[i + 1] - self.true_vertical_depth[i]
                    if delta_md > 0:
                        inc_rad = np.arccos(abs(delta_tvd) / delta_md)
                        inc_deg = np.degrees(inc_rad)
                    else:
                        inc_deg = 0.0
                else:
                    # For last point, use previous point's inclination
                    if i > 0:
                        delta_md = self.measured_depth[i] - self.measured_depth[i - 1]
                        delta_tvd = self.true_vertical_depth[i] - self.true_vertical_depth[i - 1]
                        if delta_md > 0:
                            inc_rad = np.arccos(abs(delta_tvd) / delta_md)
                            inc_deg = np.degrees(inc_rad)
                        else:
                            inc_deg = 0.0
                    else:
                        inc_deg = 0.0

                # Curve values (no conversion needed)
                curve_value = self.value[i] if not np.isnan(self.value[i]) else ''
                synt_curve_value = self.synt_curve[i] if not np.isnan(self.synt_curve[i]) else ''

                writer.writerow([
                    f'{md:.3f}',
                    f'{tvt:.3f}' if tvt != '' else '',
                    f'{inc_deg:.2f}',
                    f'{tvd:.3f}',
                    f'{vs:.3f}',
                    f'{curve_value:.3f}' if curve_value != '' else '',
                    f'{synt_curve_value:.3f}' if synt_curve_value != '' else ''
                ])

        units_text = "feet" if use_feet else "meters"
        print(f"Well data exported to {csv_path} in {units_text}")
        print(f"Total points: {len(self.measured_depth)}")