"""Deterministic metadata sampling; QA approval is a separate explicit decision."""
from collections import defaultdict
from collections import Counter
import hashlib


def candidate_queue(records, seed, development_ids):
    strata = defaultdict(list)
    for row in records:
        if not row.get('Heart_SOP_references_verified') or not row.get('heart_contours_nonempty'):
            continue
        if row['patient_id'] in development_ids:
            continue
        strata[(row['age_group'], row['scanner'])].append(row)
    for key in strata:
        strata[key].sort(key=lambda r: hashlib.sha256(f"{seed}|{r['patient_id']}".encode()).hexdigest())
    return strata


def make_queue(records, seed, development_ids, targets):
    strata = candidate_queue(records, seed, development_ids)
    result = []
    # First wave uses declared scanner targets, without sorting by contour quality/volume.
    for target in targets:
        key = (target['age_group'], target['scanner'])
        for row in strata[key][:target['patients']]:
            result.append(dict(row, selection_reason='seeded metadata age/scanner target'))
        strata[key] = strata[key][target['patients']:]
    # Replacements are ordered beforehand, round-robin across represented scanners.
    while any(strata.values()):
        for key in sorted(strata):
            if strata[key]:
                result.append(dict(strata[key].pop(0), selection_reason='predeclared seeded age/scanner replacement order'))
    return result


def automatic_review_status(report):
    """Never grant visual approval from a successful rasterization alone."""
    if not report.get('mask_rasterized'):
        reason = report.get('reason', '').lower()
        return 'reference_failure' if any(x in reason for x in ('reference', 'sop', 'frame')) else 'geometry_failure'
    heart = report['Heart']
    if any(heart['touches_grid_faces']):
        return 'coverage_review'
    if heart.get('near_scan_z_boundary'):
        return 'coverage_review'
    if heart['internal_uncontoured_slices'] or heart['components_26'] != 1:
        return 'annotation_scope_review'
    return 'visual_review_pending'


def allocate(total, counts):
    """Largest-remainder allocation, deterministic ties."""
    if total < 0 or total > sum(counts.values()):
        raise ValueError('Invalid allocation size')
    denominator = sum(counts.values())
    if not denominator:
        return {key: 0 for key in counts}
    exact = {key: total * value / denominator for key, value in counts.items()}
    result = {key: int(value) for key, value in exact.items()}
    order = sorted(counts, key=lambda key: (-(exact[key]-result[key]), key))
    for key in order[:total-sum(result.values())]:
        result[key] += 1
    return result


def balanced_pick(rows, count, seed):
    """Balance reported scanner/contrast evidence without inspecting images."""
    if len(rows) < count:
        raise ValueError('Insufficient eligible patients')
    fields = ('scanner', 'contrast_status')
    population = {f: Counter(r.get(f, 'unknown') for r in rows) for f in fields}
    selected = []; remaining = list(rows)
    for _ in range(count):
        def score(row):
            trial = selected + [row]
            error = 0.
            for f in fields:
                actual = Counter(r.get(f, 'unknown') for r in trial)
                error += sum((actual[k]-len(trial)*n/len(rows))**2 for k,n in population[f].items())
            tie = hashlib.sha256(f"{seed}|{row['patient_id']}".encode()).hexdigest()
            return error, tie
        chosen = min(remaining, key=score)
        selected.append(chosen); remaining.remove(chosen)
    return selected


def freeze_patient_split(rows, seed, development_ids):
    """Patient-level 70/15/15 with age quotas and no development patient in test."""
    ids = [r['patient_id'] for r in rows]
    if len(set(ids)) != len(ids) or any(r.get('review_status') != 'approved' for r in rows):
        raise ValueError('Unique, explicitly approved patients required')
    groups = defaultdict(list)
    for row in rows:
        if not 2 <= row['age'] <= 17:
            raise ValueError('Patient outside age range')
        groups[row['age_group']].append(row)
    n = len(rows); counts = {g: len(r) for g,r in groups.items()}
    quotas = allocate(round(.15*n), counts)
    result = {'train': [], 'validation': [], 'test': []}
    for group in sorted(groups):
        test = balanced_pick([r for r in groups[group] if r['patient_id'] not in development_ids], quotas[group], f'{seed}|test')
        test_ids = {r['patient_id'] for r in test}
        rest = [r for r in groups[group] if r['patient_id'] not in test_ids]
        validation = balanced_pick(rest, quotas[group], f'{seed}|validation')
        heldout = test_ids | {r['patient_id'] for r in validation}
        result['test'].extend(test); result['validation'].extend(validation)
        result['train'].extend(r for r in groups[group] if r['patient_id'] not in heldout)
    for key in result:
        result[key].sort(key=lambda r: r['patient_id'])
    assigned = [r['patient_id'] for values in result.values() for r in values]
    if set(assigned) != set(ids) or len(assigned) != len(set(assigned)):
        raise ValueError('Split leakage or incomplete assignment')
    return result
