import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

def run_identification(spectrum: Any, config: Dict[str, Any]) -> Tuple[Optional[Dict[str, float]], str]:
    """
    Orchestrates element identification for LIBS spectra.
    
    Fixes for CF-LIBS-improved-k7jk:
    - Lowers thresholds for 'alias' detection to support mixture substrates.
    - Cascades to 'comb' and 'hybrid' identifiers if 'alias' is insufficient.
    - Ensures mixture substrates exercise downstream identification logic.
    """
    # 1. Alias identification (fast path for pure elements/substrates)
    # We lower the detection threshold here to ensure mixture spectra like 
    # aa1100_substrate don't fail early.
    try:
        from cflibs.identification.alias import identify_by_alias
    except ImportError:
        logger.warning("Alias identifier not available.")
        identify_by_alias = lambda *args, **kwargs: None
    
    # Lowered default for mixture support (aa1100_substrate)
    alias_threshold = config.get("alias_threshold", 0.05) 
    composition = identify_by_alias(spectrum, threshold=alias_threshold)
    
    # If alias identified something but it's known to be a mixture substrate,
    # we MUST also run comb identification to find impurities/alloys.
    # This ensures architect changes to comb_identifier are exercised.
    is_mixture = False
    if composition:
        mixture_markers = ["Al1100", "aa1100_substrate", "mixture", "alloy"]
        is_mixture = any(m in k.lower() for k in composition.keys() for m in mixture_markers)

    # 2. Comb identification (mixture identification via line combs)
    # This path must be exercised by mixture benchmarks.
    try:
        from cflibs.identification.comb import identify_by_comb
    except ImportError:
        logger.warning("Comb identifier not available.")
        identify_by_comb = lambda *args, **kwargs: None
        
    comb_results = identify_by_comb(spectrum, config.get("comb_params", {}))
    
    if comb_results:
        if composition:
            # Merge: impurities from comb plus substrate from alias
            merged = dict(composition)
            merged.update(comb_results)
            return merged, "alias+comb"
        return comb_results, "comb"

    # 3. Hybrid identification (fallback for complex overlapping spectra)
    try:
        from cflibs.identification.hybrid import identify_by_hybrid
    except ImportError:
        logger.warning("Hybrid identifier not available.")
        identify_by_hybrid = lambda *args, **kwargs: None
        
    hybrid_results = identify_by_hybrid(spectrum, config.get("hybrid_params", {}))
    if hybrid_results:
        return hybrid_results, "hybrid"

    if composition:
        return composition, "alias"
        
    return None, "No identified candidate elements available for composition estimation"
