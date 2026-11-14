"""
Configuration settings, path constants, and column mappings for PipeTwin.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CLEANED_DIR = DATA_DIR / "cleaned"
PROCESSED_DIR = BASE_DIR / "processed"

# Ensure runtime directories exist
CLEANED_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Grid parameters
SEGMENT_LENGTH_M = 10.0
EPSILON_M = 1e-9

# Cyrillic to Latin transliteration dictionary for snake_case conversion
CYRILLIC_TO_LATIN = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
    'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
    'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
    'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch',
    'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
    '№': 'no', '%': 'pct'
}

# Canonical column mappings for primary analytical variables
CANONICAL_COL_MAPPINGS = {
    'passport.csv': {
        'Данные по протяжённости в структуре сети OISPIPE Паспортизация_ID простого участка трубопровода': 'id',
        'Данные по протяжённости в структуре сети OISPIPE Паспортизация_L простого участка, км': 'length_km',
        'Общие данные_D, мм': 'diameter_mm',
        'Общие данные_Толщина  стенки по исполнительной доку-ии, мм': 'nominal_thickness_mm',
        'Материал_Наименование материала': 'steel_grade',
        'Внутреннее покрытие_Тип внутреннего покрытия': 'internal_coating',
        'Общие данные_Дата ввода': 'commissioning_date',
        'Данные по протяжённости в структуре сети OISPIPE Паспортизация_ID основного участка трубопровода': 'main_pipeline_id',
        'Данные по протяжённости в структуре сети OISPIPE Паспортизация_ID направления трубопровода': 'direction_id',
        'Общие данные_Название участка по эксплуатационному паспорту': 'pipeline_name',
        'Общие данные_Длина по бух.учету, м': 'length_accounting_m',
        'Материал_Толщина  стенки отбраковочная, мм': 'scrap_wall_thickness_mm',
    },
    'profile.csv': {
        'id': 'id',
        'lon': 'lon',
        'lat': 'lat',
        'z': 'z',
        'total_distance': 'total_distance',
    },
    'uzt.csv': {
        'id': 'id',
        'Номер контролируемого участка': 'control_section_num',
        'start_distance': 'start_distance',
        'ID контрольной точки': 'control_point_id',
        'Номер точки': 'point_num',
        'Адрес от начала участка_1': 'distance',
        'date': 'date',
        'ID замера': 'measurement_id',
        'angle': 'angle',
        'residual_wall_thickness': 'residual_wall_thickness',
    },
    'vtd.csv': {
        'row_id': 'row_id',
        'id': 'id',
        'date': 'date',
        'vtd_date_raw': 'vtd_date_raw',
        'Distance': 'distance',
        'Distance_raw': 'distance_raw',
        'accumulated_value': 'accumulated_value',
        'section_length': 'section_length',
        'is_beyond_pot': 'is_beyond_pot',
        'Section Number': 'section_number',
        'Feature Number': 'feature_number',
        'Feature Description': 'feature_description',
        'Defect Type': 'defect_type',
        'trait': 'trait',
        'Wall Thickness': 'wall_thickness',
        'Residual Wall Thickness': 'residual_wall_thickness',
        'Defect Depth in %': 'defect_depth_pct',
        'Defect Depth': 'defect_depth',
        'Angle': 'angle',
        'Feature Length': 'defect_length',
        'Feature Width': 'defect_width',
    },
    'regime.csv': {
        'Simple Section ID': 'id',
        'Year': 'year',
        'Sample Installation Date': 'sample_installation_date',
        'Sample Removal Date': 'sample_removal_date',
        'UKK Number': 'ukk_number',
        'UKK Address from the start of the simple section': 'ukk_distance',
        'Length': 'length',
        'D': 'diameter',
        'Wall Thickness': 'wall_thickness',
        'Qzh': 'flow_liquid',
        'H2O': 'watercut_pct',
        'Exposure Time, days': 'exposure_days',
    },
    'tech_mode_for_embedings_with_length_with_temperatures_ids16.csv': {
        'simple_section_id': 'id',
        'calc_date': 'date',
        'p_start_fact': 'p_start_fact',
        'p_end_fact': 'p_end_fact',
        'p_start': 'p_start',
        'p_end': 'p_end',
        'q_oil': 'q_oil',
        'q': 'q_liquid',
        'watercut': 'watercut_pct',
        'gas_content_rate': 'gas_content_rate',
        'v_liquid_true': 'v_liquid_true',
        'flow_regime_real': 'flow_regime',
        'fill_coefficient': 'fill_coefficient',
        'q_gas': 'q_gas',
        'density_liquid_work': 'density_liquid_work',
        'viscosity_liquid_work': 'viscosity_liquid_work',
        'density_oil': 'density_oil',
        'pit_corrosion_intensity': 'pit_corrosion_intensity',
        'v_mix': 'v_mix',
        't_mix': 't_mix',
        'length': 'length',
        'd_mm': 'diameter_mm',
        't_start': 't_start',
        't_end': 't_end',
    }
}
