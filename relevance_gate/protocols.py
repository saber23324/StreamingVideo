"""Keep weak pair-construction targets separate from semantic ground truth."""


TRUSTED = "visible_correspondence"
WEAK = "weak_pairing"


def target_rows(rows, protocol=TRUSTED):
    if protocol not in (TRUSTED, WEAK):
        raise ValueError("unknown label protocol")
    if protocol == TRUSTED:
        return [{**row, "label_protocol": TRUSTED} for row in rows]
    result = []
    for row in rows:
        target = row.get("pairing_target")
        if target is None:
            continue
        if type(target) is not int or target not in (0, 1) or not row.get("pairing_target_provenance"):
            raise ValueError("weak pairing targets require explicit binary value and construction provenance")
        result.append({**row, "semantic_visible_correspondence": row["visible_correspondence"],
                       "semantic_label_status": row.get("label_status"), "semantic_label_provenance": row.get("label_provenance"),
                       "visible_correspondence": target,
                       "label_protocol": WEAK, "label_status": "weak_pairing_only",
                       "label_provenance": row["pairing_target_provenance"]})
    return result


def protocol_scope(protocol):
    return ("weak pair-construction targets only; manual semantic review pending; not visible-correspondence validation"
            if protocol == WEAK else "trusted visible-correspondence labels only; unknowns counted separately")
