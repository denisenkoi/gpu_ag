import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DeltaAnalyzer:
    """Analyzes deltas between reference and generated data using DataFrames with detailed statistics"""
    
    def __init__(self, tolerance: float = 0.001):
        """Initialize delta analyzer
        
        Args:
            tolerance: Tolerance for considering deltas significant
        """
        self.tolerance = tolerance
        self.delta_dataframes = {}
        
    def create_all_delta_dataframes(self, reference: Dict, generated: Dict) -> Dict[str, pd.DataFrame]:
        """Create delta DataFrames for all data types
        
        Args:
            reference: Reference data (already interpolated)
            generated: Generated data
            
        Returns:
            Dictionary of DataFrames with deltas
        """
        logger.info("Creating delta DataFrames for analysis")
        
        # 1. Trajectory deltas
        ref_trajectory = self._get_nested_value(reference, ['well', 'points'])
        gen_trajectory = self._get_nested_value(generated, ['well', 'points'])
        if ref_trajectory and gen_trajectory:
            self.delta_dataframes['trajectory'] = self._create_trajectory_deltas_df(
                ref_trajectory, gen_trajectory
            )
        
        # 2. WellLog points deltas
        ref_welllog_points = self._get_nested_value(reference, ['wellLog', 'points'])
        gen_welllog_points = self._get_nested_value(generated, ['wellLog', 'points'])
        if ref_welllog_points and gen_welllog_points:
            self.delta_dataframes['welllog_points'] = self._create_welllog_points_deltas_df(
                ref_welllog_points, gen_welllog_points
            )
        
        # 3. WellLog tvdSortedPoints deltas
        ref_welllog_tvd = self._get_nested_value(reference, ['wellLog', 'tvdSortedPoints'])
        gen_welllog_tvd = self._get_nested_value(generated, ['wellLog', 'tvdSortedPoints'])
        if ref_welllog_tvd and gen_welllog_tvd:
            self.delta_dataframes['welllog_tvd'] = self._create_welllog_tvd_deltas_df(
                ref_welllog_tvd, gen_welllog_tvd
            )
        
        # 4. TypeLog points deltas
        ref_typewell_points = self._get_nested_value(reference, ['typeLog', 'points'])
        gen_typewell_points = self._get_nested_value(generated, ['typeLog', 'points'])
        if ref_typewell_points and gen_typewell_points:
            self.delta_dataframes['typewell_points'] = self._create_typewell_points_deltas_df(
                ref_typewell_points, gen_typewell_points
            )
        
        # 5. TypeLog tvdSortedPoints deltas
        ref_typewell_tvd = self._get_nested_value(reference, ['typeLog', 'tvdSortedPoints'])
        gen_typewell_tvd = self._get_nested_value(generated, ['typeLog', 'tvdSortedPoints'])
        if ref_typewell_tvd and gen_typewell_tvd:
            self.delta_dataframes['typewell_tvd'] = self._create_typewell_tvd_deltas_df(
                ref_typewell_tvd, gen_typewell_tvd
            )
        
        # 6. GridSlice points deltas
        ref_grid_points = self._get_nested_value(reference, ['gridSlice', 'points'])
        gen_grid_points = self._get_nested_value(generated, ['gridSlice', 'points'])
        if ref_grid_points and gen_grid_points:
            self.delta_dataframes['grid_slice'] = self._create_grid_deltas_df(
                ref_grid_points, gen_grid_points
            )
        
        return self.delta_dataframes
    
    def _log_detailed_comparison_stats(self, name: str, ref_points: List, gen_points: List, 
                                      merged_df: pd.DataFrame, result_df: pd.DataFrame, 
                                      value_columns: List[str]):
        """Log detailed comparison statistics for any data type"""
        
        # Extract MD ranges
        ref_md = [p['measuredDepth'] for p in ref_points]
        gen_md = [p['measuredDepth'] for p in gen_points]
        
        ref_md_min, ref_md_max = min(ref_md), max(ref_md)
        gen_md_min, gen_md_max = min(gen_md), max(gen_md)
        
        # Coverage calculation
        if not merged_df.empty:
            compared_md = merged_df['measuredDepth'].values
            compared_min, compared_max = compared_md.min(), compared_md.max()
            compared_range = compared_max - compared_min
            ref_range = ref_md_max - ref_md_min
            gen_range = gen_md_max - gen_md_min
            ref_coverage = (compared_range / ref_range * 100) if ref_range > 0 else 0
            gen_coverage = (compared_range / gen_range * 100) if gen_range > 0 else 0
        else:
            compared_min = compared_max = 0
            ref_coverage = gen_coverage = 0
        
        # Calculate RMS and max deltas for each value column
        delta_stats = {}
        for col in value_columns:
            delta_col = f'delta_{col}'
            if delta_col in merged_df.columns:
                deltas = merged_df[delta_col].values
                delta_stats[col] = {
                    'max': np.abs(deltas).max(),
                    'rms': np.sqrt(np.mean(deltas**2)),
                    'mean': np.mean(deltas),
                    'std': np.std(deltas)
                }
        
        # Range overlap analysis
        ref_range_str = f"{ref_md_min:.1f}→{ref_md_max:.1f}"
        gen_range_str = f"{gen_md_min:.1f}→{gen_md_max:.1f}"
        
        range_mismatch = (abs(ref_md_min - gen_md_min) > 1.0 or 
                         abs(ref_md_max - gen_md_max) > 1.0)
        
        # Log comprehensive statistics
        logger.info(f"\n📊 {name} DETAILED COMPARISON STATS:")
        logger.info(f"  Reference: {len(ref_points)} points, MD range: {ref_range_str}")
        logger.info(f"  Generated: {len(gen_points)} points, MD range: {gen_range_str}")
        
        if range_mismatch:
            logger.warning(f"  ❌ RANGE MISMATCH! Different MD ranges detected")
            if ref_md_min != gen_md_min:
                logger.warning(f"    Start difference: {abs(ref_md_min - gen_md_min):.1f} ft")
            if ref_md_max != gen_md_max:
                logger.warning(f"    End difference: {abs(ref_md_max - gen_md_max):.1f} ft")
        
        if not merged_df.empty:
            logger.info(f"  Compared: {len(merged_df)} points, MD range: {compared_min:.1f}→{compared_max:.1f}")
            logger.info(f"  Coverage: Ref={ref_coverage:.1f}%, Gen={gen_coverage:.1f}%")
            
            # Log delta statistics
            for col, stats in delta_stats.items():
                logger.info(f"  {col.upper()} deltas: max={stats['max']:.6f}, rms={stats['rms']:.6f}, "
                           f"mean={stats['mean']:.6f}, std={stats['std']:.6f}")
        else:
            logger.warning(f"  ❌ NO OVERLAP! Cannot compare - no common MD values")
            
        logger.info(f"  Significant deltas (>{self.tolerance}): {len(result_df)}")
        
        # Coverage warnings
        if ref_coverage < 90:
            logger.warning(f"  ⚠️ LOW REFERENCE COVERAGE: Only {ref_coverage:.1f}% of reference data compared")
        if gen_coverage < 90:
            logger.warning(f"  ⚠️ LOW GENERATED COVERAGE: Only {gen_coverage:.1f}% of generated data compared")
    
    def _create_trajectory_deltas_df(self, ref_points: List, gen_points: List) -> pd.DataFrame:
        """Create DataFrame with trajectory deltas with detailed stats"""
        
        # Convert to DataFrames
        ref_df = pd.DataFrame(ref_points)
        gen_df = pd.DataFrame(gen_points)
        
        # Merge on MD (should match exactly after interpolation)
        merged_df = ref_df.merge(gen_df, on='measuredDepth', how='inner', suffixes=('_ref', '_gen'))
        
        if merged_df.empty:
            logger.warning("No MD matches found between reference and generated trajectory")
            return pd.DataFrame()
        
        # Calculate deltas
        merged_df['delta_md'] = 0.0  # Should be zero after interpolation
        merged_df['delta_tvd'] = merged_df['trueVerticalDepth_gen'] - merged_df['trueVerticalDepth_ref']
        merged_df['delta_ns'] = merged_df['northSouth_gen'] - merged_df['northSouth_ref']
        merged_df['delta_ew'] = merged_df['eastWest_gen'] - merged_df['eastWest_ref']
        
        # Handle optional fields
        if 'inclinationRad_ref' in merged_df.columns and 'inclinationRad_gen' in merged_df.columns:
            merged_df['delta_incl'] = merged_df['inclinationRad_gen'] - merged_df['inclinationRad_ref']
        if 'azimutRad_ref' in merged_df.columns and 'azimutRad_gen' in merged_df.columns:
            merged_df['delta_azim'] = merged_df['azimutRad_gen'] - merged_df['azimutRad_ref']
        
        # Calculate total absolute delta
        delta_cols = ['delta_tvd', 'delta_ns', 'delta_ew']
        if 'delta_incl' in merged_df.columns:
            delta_cols.append('delta_incl')
        if 'delta_azim' in merged_df.columns:
            delta_cols.append('delta_azim')
        
        merged_df['abs_delta_total'] = merged_df[delta_cols].abs().sum(axis=1)
        
        # Filter only significant deltas
        significant_mask = merged_df['abs_delta_total'] > self.tolerance
        result_df = merged_df[significant_mask].copy()
        
        # Log detailed statistics
        value_cols = ['tvd', 'ns', 'ew']
        if 'inclinationRad_ref' in merged_df.columns:
            value_cols.append('incl')
        if 'azimutRad_ref' in merged_df.columns:
            value_cols.append('azim')
        self._log_detailed_comparison_stats("TRAJECTORY", ref_points, gen_points, 
                                           merged_df, result_df, value_cols)
        
        # Rename columns for consistency
        result_df = result_df.rename(columns={
            'measuredDepth': 'md',
            'trueVerticalDepth_ref': 'ref_tvd',
            'trueVerticalDepth_gen': 'gen_tvd',
            'northSouth_ref': 'ref_ns',
            'northSouth_gen': 'gen_ns',
            'eastWest_ref': 'ref_ew',
            'eastWest_gen': 'gen_ew'
        })
        
        return result_df
    
    def _create_welllog_points_deltas_df(self, ref_points: List, gen_points: List) -> pd.DataFrame:
        """Create DataFrame with wellLog points deltas with detailed stats"""
        
        ref_df = pd.DataFrame(ref_points)
        gen_df = pd.DataFrame(gen_points)
        
        # Merge on MD
        merged_df = ref_df.merge(gen_df, on='measuredDepth', how='inner', suffixes=('_ref', '_gen'))
        
        if merged_df.empty:
            logger.warning("No MD matches found between reference and generated wellLog points")
            return pd.DataFrame()
        
        # Calculate deltas
        merged_df['delta_md'] = 0.0
        merged_df['delta_data'] = merged_df['data_gen'] - merged_df['data_ref']
        merged_df['abs_delta_total'] = merged_df['delta_data'].abs()
        
        # Filter significant deltas
        significant_mask = merged_df['abs_delta_total'] > self.tolerance
        result_df = merged_df[significant_mask].copy()
        
        # Log detailed statistics
        self._log_detailed_comparison_stats("WELLLOG_POINTS", ref_points, gen_points, 
                                           merged_df, result_df, ['data'])
        
        # Rename columns
        result_df = result_df.rename(columns={
            'measuredDepth': 'md',
            'data_ref': 'ref_data',
            'data_gen': 'gen_data'
        })
        
        return result_df
    
    def _create_welllog_tvd_deltas_df(self, ref_points: List, gen_points: List) -> pd.DataFrame:
        """Create DataFrame with wellLog tvdSortedPoints deltas with fake TVD comparison and detailed stats"""
        
        ref_df = pd.DataFrame(ref_points)
        gen_df = pd.DataFrame(gen_points)
        
        # ========== FAKE TVD COMPARISON ==========
        # Create fake generated data with TVD = MD
        fake_gen_points = []
        for point in gen_points:
            fake_point = point.copy()
            fake_point['trueVerticalDepth'] = fake_point['measuredDepth']  # fakeTVD = MD
            fake_gen_points.append(fake_point)
        
        fake_gen_df = pd.DataFrame(fake_gen_points)
        
        # Compare reference vs fake generated
        fake_merged_df = ref_df.merge(fake_gen_df, on='measuredDepth', how='inner', suffixes=('_ref', '_fake'))
        if not fake_merged_df.empty:
            fake_merged_df['fake_delta_tvd'] = fake_merged_df['trueVerticalDepth_fake'] - fake_merged_df['trueVerticalDepth_ref']
            fake_merged_df['fake_delta_data'] = fake_merged_df['data_fake'] - fake_merged_df['data_ref']
            fake_merged_df['fake_abs_delta_total'] = fake_merged_df['fake_delta_tvd'].abs() + fake_merged_df['fake_delta_data'].abs()
            
            fake_significant_count = len(fake_merged_df[fake_merged_df['fake_abs_delta_total'] > self.tolerance])
            logger.info(f"🔍 WellLog TVD FAKE comparison (fakeTVD = MD): {fake_significant_count} differences")
        else:
            logger.warning("No MD matches found for fake TVD comparison")
        
        # ========== REAL TVD COMPARISON ==========
        # Merge on MD
        merged_df = ref_df.merge(gen_df, on='measuredDepth', how='inner', suffixes=('_ref', '_gen'))
        
        if merged_df.empty:
            logger.warning("No MD matches found between reference and generated wellLog TVD points")
            return pd.DataFrame()
        
        # Calculate deltas
        merged_df['delta_md'] = 0.0
        merged_df['delta_tvd'] = merged_df['trueVerticalDepth_gen'] - merged_df['trueVerticalDepth_ref']
        merged_df['delta_data'] = merged_df['data_gen'] - merged_df['data_ref']
        merged_df['abs_delta_total'] = merged_df['delta_tvd'].abs() + merged_df['delta_data'].abs()
        
        # Filter significant deltas
        significant_mask = merged_df['abs_delta_total'] > self.tolerance
        result_df = merged_df[significant_mask].copy()
        
        # Log detailed statistics
        self._log_detailed_comparison_stats("WELLLOG_TVD", ref_points, gen_points, 
                                           merged_df, result_df, ['tvd', 'data'])
        
        logger.info(f"🔍 WellLog TVD REAL comparison (real TVD): {len(result_df)} differences")
        
        # Rename columns
        result_df = result_df.rename(columns={
            'measuredDepth': 'md',
            'trueVerticalDepth_ref': 'ref_tvd',
            'trueVerticalDepth_gen': 'gen_tvd',
            'data_ref': 'ref_data',
            'data_gen': 'gen_data'
        })
        
        return result_df
    
    def _create_typewell_points_deltas_df(self, ref_points: List, gen_points: List) -> pd.DataFrame:
        """Create DataFrame with typeLog points deltas with detailed stats"""
        
        ref_df = pd.DataFrame(ref_points)
        gen_df = pd.DataFrame(gen_points)
        
        # Merge on MD
        merged_df = ref_df.merge(gen_df, on='measuredDepth', how='inner', suffixes=('_ref', '_gen'))
        
        if merged_df.empty:
            logger.warning("No MD matches found between reference and generated typewell points")
            return pd.DataFrame()
        
        # Calculate deltas
        merged_df['delta_md'] = 0.0
        merged_df['delta_data'] = merged_df['data_gen'] - merged_df['data_ref']
        merged_df['abs_delta_total'] = merged_df['delta_data'].abs()
        
        # Filter significant deltas
        significant_mask = merged_df['abs_delta_total'] > self.tolerance
        result_df = merged_df[significant_mask].copy()
        
        # Log detailed statistics
        self._log_detailed_comparison_stats("TYPEWELL_POINTS", ref_points, gen_points, 
                                           merged_df, result_df, ['data'])
        
        # Rename columns
        result_df = result_df.rename(columns={
            'measuredDepth': 'md',
            'data_ref': 'ref_data',
            'data_gen': 'gen_data'
        })
        
        return result_df
    
    def _create_typewell_tvd_deltas_df(self, ref_points: List, gen_points: List) -> pd.DataFrame:
        """Create DataFrame with typeLog tvdSortedPoints deltas with fake TVD comparison and detailed stats"""
        
        ref_df = pd.DataFrame(ref_points)
        gen_df = pd.DataFrame(gen_points)
        
        # ========== FAKE TVD COMPARISON ==========
        # Create fake generated data with TVD = MD
        fake_gen_points = []
        for point in gen_points:
            fake_point = point.copy()
            fake_point['trueVerticalDepth'] = fake_point['measuredDepth']  # fakeTVD = MD
            fake_gen_points.append(fake_point)
        
        fake_gen_df = pd.DataFrame(fake_gen_points)
        
        # Compare reference vs fake generated
        fake_merged_df = ref_df.merge(fake_gen_df, on='measuredDepth', how='inner', suffixes=('_ref', '_fake'))
        if not fake_merged_df.empty:
            fake_merged_df['fake_delta_tvd'] = fake_merged_df['trueVerticalDepth_fake'] - fake_merged_df['trueVerticalDepth_ref']
            fake_merged_df['fake_delta_data'] = fake_merged_df['data_fake'] - fake_merged_df['data_ref']
            fake_merged_df['fake_abs_delta_total'] = fake_merged_df['fake_delta_tvd'].abs() + fake_merged_df['fake_delta_data'].abs()
            
            fake_significant_count = len(fake_merged_df[fake_merged_df['fake_abs_delta_total'] > self.tolerance])
            logger.info(f"🔍 TypeLog TVD FAKE comparison (fakeTVD = MD): {fake_significant_count} differences")
        else:
            logger.warning("No MD matches found for fake typewell TVD comparison")
        
        # ========== REAL TVD COMPARISON ==========
        # Merge on MD
        merged_df = ref_df.merge(gen_df, on='measuredDepth', how='inner', suffixes=('_ref', '_gen'))
        
        if merged_df.empty:
            logger.warning("No MD matches found between reference and generated typewell TVD points")
            return pd.DataFrame()
        
        # Calculate deltas
        merged_df['delta_md'] = 0.0
        merged_df['delta_tvd'] = merged_df['trueVerticalDepth_gen'] - merged_df['trueVerticalDepth_ref']
        merged_df['delta_data'] = merged_df['data_gen'] - merged_df['data_ref']
        merged_df['abs_delta_total'] = merged_df['delta_tvd'].abs() + merged_df['delta_data'].abs()
        
        # Filter significant deltas
        significant_mask = merged_df['abs_delta_total'] > self.tolerance
        result_df = merged_df[significant_mask].copy()
        
        # Log detailed statistics
        self._log_detailed_comparison_stats("TYPEWELL_TVD", ref_points, gen_points, 
                                           merged_df, result_df, ['tvd', 'data'])
        
        logger.info(f"🔍 TypeLog TVD REAL comparison (real TVD): {len(result_df)} differences")
        
        # Rename columns
        result_df = result_df.rename(columns={
            'measuredDepth': 'md',
            'trueVerticalDepth_ref': 'ref_tvd',
            'trueVerticalDepth_gen': 'gen_tvd',
            'data_ref': 'ref_data',
            'data_gen': 'gen_data'
        })
        
        return result_df
    
    def _create_grid_deltas_df(self, ref_points: List, gen_points: List) -> pd.DataFrame:
        """Create DataFrame with gridSlice points deltas with detailed stats"""
        
        ref_df = pd.DataFrame(ref_points)
        gen_df = pd.DataFrame(gen_points)
        
        # Merge on MD
        merged_df = ref_df.merge(gen_df, on='measuredDepth', how='inner', suffixes=('_ref', '_gen'))
        
        if merged_df.empty:
            logger.warning("No MD matches found between reference and generated grid points")
            return pd.DataFrame()
        
        # Calculate deltas
        merged_df['delta_md'] = 0.0
        
        # Handle different field names for TVDSS
        tvdss_ref_col = 'trueVerticalDepthSubSea_ref' if 'trueVerticalDepthSubSea_ref' in merged_df.columns else 'trueVerticalDepth_ref'
        tvdss_gen_col = 'trueVerticalDepthSubSea_gen' if 'trueVerticalDepthSubSea_gen' in merged_df.columns else 'trueVerticalDepth_gen'
        
        merged_df['delta_tvdss'] = merged_df[tvdss_gen_col] - merged_df[tvdss_ref_col]
        merged_df['delta_ns'] = merged_df['northSouth_gen'] - merged_df['northSouth_ref']
        merged_df['delta_ew'] = merged_df['eastWest_gen'] - merged_df['eastWest_ref']
        merged_df['delta_vs'] = merged_df['verticalSection_gen'] - merged_df['verticalSection_ref']
        
        # Calculate total absolute delta
        delta_cols = ['delta_tvdss', 'delta_ns', 'delta_ew', 'delta_vs']
        merged_df['abs_delta_total'] = merged_df[delta_cols].abs().sum(axis=1)
        
        # Filter significant deltas
        significant_mask = merged_df['abs_delta_total'] > self.tolerance
        result_df = merged_df[significant_mask].copy()
        
        # Log detailed statistics
        self._log_detailed_comparison_stats("GRID_SLICE", ref_points, gen_points, 
                                           merged_df, result_df, ['tvdss', 'ns', 'ew', 'vs'])
        
        # Rename columns
        rename_dict = {
            'measuredDepth': 'md',
            'northSouth_ref': 'ref_ns',
            'northSouth_gen': 'gen_ns',
            'eastWest_ref': 'ref_ew',
            'eastWest_gen': 'gen_ew',
            'verticalSection_ref': 'ref_vs',
            'verticalSection_gen': 'gen_vs'
        }
        rename_dict[tvdss_ref_col] = 'ref_tvdss'
        rename_dict[tvdss_gen_col] = 'gen_tvdss'
        
        result_df = result_df.rename(columns=rename_dict)
        
        return result_df
    
    def log_detailed_analysis(self):
        """Log detailed analysis of all delta DataFrames"""
        logger.info("\n" + "="*80)
        logger.info("DETAILED DATAFRAMES DELTA ANALYSIS")
        logger.info("="*80)
        
        analysis_names = {
            'trajectory': ('TRAJECTORY', ['delta_md', 'delta_tvd', 'delta_ns', 'delta_ew']),
            'welllog_points': ('WELLLOG_POINTS', ['delta_md', 'delta_data']),
            'welllog_tvd': ('WELLLOG_TVD', ['delta_md', 'delta_tvd', 'delta_data']),
            'typewell_points': ('TYPEWELL_POINTS', ['delta_md', 'delta_data']),
            'typewell_tvd': ('TYPEWELL_TVD', ['delta_md', 'delta_tvd', 'delta_data']),
            'grid_slice': ('GRID_SLICE', ['delta_md', 'delta_tvdss', 'delta_ns', 'delta_ew', 'delta_vs'])
        }
        
        for key, (name, delta_cols) in analysis_names.items():
            if key in self.delta_dataframes and not self.delta_dataframes[key].empty:
                self._log_single_dataframe_analysis(name, self.delta_dataframes[key], delta_cols)
            else:
                logger.info(f"📊 {name}: No significant deltas found")
        
        logger.info("="*80)
    
    def _log_single_dataframe_analysis(self, name: str, df: pd.DataFrame, delta_columns: List[str]):
        """Log detailed analysis of a single DataFrame"""
        logger.info(f"\n📊 {name} DELTAS ANALYSIS:")
        logger.info(f"  Total rows with significant deltas: {len(df)}")
        
        if len(df) == 0:
            return
        
        # MD range analysis
        if 'md' in df.columns:
            logger.info(f"  MD range: {df['md'].min():.1f}→{df['md'].max():.1f}")
        
        # Delta statistics for each column
        for col in delta_columns:
            if col in df.columns:
                col_data = df[col]
                logger.info(f"  {col}: min={col_data.min():.6f}, max={col_data.max():.6f}, "
                           f"mean={col_data.mean():.6f}, std={col_data.std():.6f}")
        
        # Show top deltas
        if 'abs_delta_total' in df.columns and len(df) > 0:
            top_count = min(5, len(df))
            top_deltas = df.nlargest(top_count, 'abs_delta_total')
            logger.info(f"  TOP {top_count} LARGEST DELTAS:")
            for idx, row in top_deltas.iterrows():
                md_info = f"MD={row['md']:.1f}" if 'md' in row else ""
                delta_info = ", ".join([f"{col}={row[col]:.6f}" for col in delta_columns if col in row])
                logger.info(f"    {md_info}: {delta_info}")
    
    def _get_nested_value(self, data: Dict, path: List[str]) -> Optional[List]:
        """Get nested value from dictionary using path"""
        current = data
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current