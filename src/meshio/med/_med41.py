import numpy as np

"Ce code gère les champs de données dans les fichiers MED, en utilisant des masques de bits pour indiquer les types d'entités et les types géométriques associés. "
"Il fournit des fonctions pour décoder ces masques et une classe pour construire ces masques lors de l'écriture des données de champ."
"L'idée : au lieu de stocker une liste de chaînes de caractères, on stocke un seul entier 32 bits où chaque bit représente la présence/absence d'un élément" 


_ATTR_ENTITY_MASK = "LEN"
_ATTR_ENTITY_ALL  = "LAA" # Nombre de pas de temps/itérations où tous les types d'entités sont présents

_ATTR_GEO = {
    "MED_CELL":            ("LGC", "LCA"),  
    "MED_DESCENDING_FACE": ("LGF", "LFA"),
    "MED_DESCENDING_EDGE": ("LGE", "LEA"),
    "MED_NODE":            ("LGN", "LNA"),
    "MED_NODE_ELEMENT":    ("LGT", "LTA"),
    "MED_STRUCT_ELEMENT":  ("LGS", "LSA"),
}

_ENTITY_BIT = {
    "MED_CELL":            0,
    "MED_DESCENDING_FACE": 1,
    "MED_DESCENDING_EDGE": 2,
    "MED_NODE":            3,
    "MED_NODE_ELEMENT":    4,
    "MED_STRUCT_ELEMENT":  5,
}
_BIT_TO_ENTITY = {v: k for k, v in _ENTITY_BIT.items()}

_GEO_ORDER = { 
    "MED_CELL": [
        "MED_POINT1", "MED_SEG2", "MED_SEG3", "MED_SEG4", 
        "MED_TRIA3", "MED_QUAD4", "MED_TRIA6", "MED_TRIA7", 
        "MED_QUAD8", "MED_QUAD9", "MED_TETRA4", "MED_PYRA5", 
        "MED_PENTA6", "MED_HEXA8", "MED_TETRA10", "MED_OCTA12", 
        "MED_PYRA13", "MED_PENTA15", "MED_PENTA18", "MED_HEXA20", 
        "MED_HEXA27", "MED_POLYGON", "MED_POLYGON2", "MED_POLYHEDRON",
    ],
    "MED_DESCENDING_FACE": [
        "MED_TRIA3", "MED_QUAD4", "MED_TRIA6", "MED_TRIA7", 
        "MED_QUAD8", "MED_QUAD9", "MED_POLYGON", "MED_POLYGON2",
    ],
    "MED_DESCENDING_EDGE": ["MED_SEG2", "MED_SEG3", "MED_SEG4"],
    "MED_NODE": ["MED_NO_GEOTYPE"],
    "MED_NODE_ELEMENT": [
        "MED_POINT1", "MED_SEG2", "MED_SEG3", "MED_SEG4", 
        "MED_TRIA3", "MED_QUAD4", "MED_TRIA6", "MED_TRIA7", 
        "MED_QUAD8", "MED_QUAD9", "MED_TETRA4", "MED_PYRA5", 
        "MED_PENTA6", "MED_HEXA8", "MED_TETRA10", "MED_OCTA12", 
        "MED_PYRA13", "MED_PENTA15", "MED_PENTA18", "MED_HEXA20", 
        "MED_HEXA27", "MED_POLYGON", "MED_POLYGON2", "MED_POLYHEDRON",
    ],
    "MED_STRUCT_ELEMENT": [],
}

def _bit_set(mask: np.uint32, pos: int) -> np.uint32:
    return np.uint32(int(mask) | (1 << pos))

def _bit_test(mask: np.uint32, pos: int) -> bool:
    return bool(int(mask) & (1 << pos))

def decode_entity_mask(mask: np.uint32) -> list[str]:
    return [_BIT_TO_ENTITY[b] for b in range(6) if _bit_test(mask, b)]

def decode_geo_mask(entity_type: str, mask: np.uint32) -> list[str]:
    order = _GEO_ORDER.get(entity_type, [])
    return [order[b] for b in range(len(order)) if _bit_test(mask, b)]

def _read_u32(grp, attr_name) -> np.uint32 | None:
    if attr_name not in grp.attrs:
        return None
    return np.uint32(int(grp.attrs[attr_name]))

def read_field_types(field_grp, numdt=None, numit=None) -> dict | None:
    target = field_grp
    len_mask = _read_u32(target, _ATTR_ENTITY_MASK)
    if len_mask is None:
        return None

    result = {}
    for et in decode_entity_mask(len_mask):
        geo_attr, all_attr = _ATTR_GEO[et]
        geo_mask = _read_u32(target, geo_attr)
        geo_types = decode_geo_mask(et, geo_mask) if geo_mask is not None else []
        usedbyncs = int(field_grp.attrs[all_attr]) if all_attr in field_grp.attrs else 0
        result[et] = {"geo_types": geo_types, "usedbyncs": usedbyncs}
    return result

class FieldBitmaskWriter:
    def __init__(self):
        self._g_entity : np.uint32         = np.uint32(0) # Masque global des types d'entités présents dans le champ
        self._g_geo    : dict[str,np.uint32] = {} # Masque global des types géométriques présents pour chaque type d'entité
        self._s_entity : dict[str,np.uint32]         = {} # Masque des types d'entités présents pour chaque pas de temps/itération
        self._s_geo    : dict[str,dict[str,np.uint32]] = {} # Masque des types géométriques présents pour chaque type d'entité et chaque pas de temps/itération

    def notify(self, entity_type: str, geo_type: str, step: str):
        ebit = _ENTITY_BIT[entity_type]
        order = _GEO_ORDER.get(entity_type, [])
        gbit  = order.index(geo_type) if geo_type in order else None

        self._g_entity = _bit_set(self._g_entity, ebit)
        if gbit is not None:
            self._g_geo.setdefault(entity_type, np.uint32(0))
            self._g_geo[entity_type] = _bit_set(self._g_geo[entity_type], gbit)

        self._s_entity.setdefault(step, np.uint32(0))
        self._s_entity[step] = _bit_set(self._s_entity[step], ebit)
        if gbit is not None:
            self._s_geo.setdefault(step, {})
            self._s_geo[step].setdefault(entity_type, np.uint32(0))
            self._s_geo[step][entity_type] = _bit_set(self._s_geo[step][entity_type], gbit)

    def flush(self, field_grp):
        def _w32(grp, name, val: np.uint32):
            grp.attrs.create(name, np.int32(val), dtype=np.dtype(">i4"))
        def _wint(grp, name, val: int):
            grp.attrs.create(name, np.int64(val))

        _w32(field_grp, _ATTR_ENTITY_MASK, self._g_entity)

        for et, gmask in self._g_geo.items():
            geo_attr, all_attr = _ATTR_GEO[et]
            _w32(field_grp, geo_attr, gmask)
            same_count = sum(1 for s, sgeo in self._s_geo.items() if sgeo.get(et) == gmask)
            _wint(field_grp, all_attr, same_count)

        same_entity_count = sum(1 for s, smask in self._s_entity.items() if smask == self._g_entity)
        _wint(field_grp, _ATTR_ENTITY_ALL, same_entity_count)

        for step, emask in self._s_entity.items():
            if step not in field_grp: continue
            sg = field_grp[step]
            _w32(sg, _ATTR_ENTITY_MASK, emask)
            for et, gmask in self._s_geo.get(step, {}).items():
                _w32(sg, _ATTR_GEO[et][0], gmask)

def _step_name(numdt: int, numit: int) -> str:
    return f"{numdt:+011d}{numit:+011d}"
