import copy
from collections import Counter
import pytest
from heart3d.cohort import make_queue, automatic_review_status, freeze_patient_split, allocate


def records():
    return [dict(patient_id=f'p{i}',age_group='2-5',scanner='scanner',
                 Heart_SOP_references_verified=True,heart_contours_nonempty=True,heart_contours=i)
            for i in range(12)]


def test_selection_independent_of_mask_size_and_excludes_development():
    rows=records(); target=[dict(age_group='2-5',scanner='scanner',patients=4)]
    original=make_queue(rows,20261006,{'p0'},target)
    changed=copy.deepcopy(rows)
    for r in changed:r['heart_contours']=1000-r['heart_contours'];r['visual_beauty']='changed'
    again=make_queue(changed,20261006,{'p0'},target)
    assert [r['patient_id'] for r in original]==[r['patient_id'] for r in again]
    assert len(original)==11 and 'p0' not in {r['patient_id'] for r in original}
    assert len({r['patient_id'] for r in original})==11


def test_metadata_failures_excluded():
    rows=records();rows[2]['Heart_SOP_references_verified']=False;rows[3]['heart_contours_nonempty']=False
    q=make_queue(rows,42,set(),[dict(age_group='2-5',scanner='scanner',patients=4)])
    assert {'p2','p3'}.isdisjoint(r['patient_id'] for r in q)


def test_technical_success_does_not_auto_approve():
    report={'mask_rasterized':True,'Heart':{'touches_grid_faces':[False]*6,
            'near_scan_z_boundary':False,'internal_uncontoured_slices':[],'components_26':1}}
    assert automatic_review_status(report)=='visual_review_pending'
    report['Heart']['near_scan_z_boundary']=True
    assert automatic_review_status(report)=='coverage_review'
    report['Heart']['near_scan_z_boundary']=False;report['Heart']['internal_uncontoured_slices']=[8]
    assert automatic_review_status(report)=='annotation_scope_review'


def test_reference_failure_classified_separately():
    assert automatic_review_status({'mask_rasterized':False,'reason':'missing SOP reference'})=='reference_failure'
    assert automatic_review_status({'mask_rasterized':False,'reason':'irregular spacing'})=='geometry_failure'


def approved_rows():
    return [dict(patient_id=f'{g}-{i}',age_group=g,age=age,scanner=f'scanner{i%3}',
        contrast_status='unknown' if i%4==0 else 'contrast_agent_reported',review_status='approved')
        for g,age in [('2-5',3),('6-11',8),('12-17',15)] for i in range(20)]


def test_frozen_split_patient_isolation_age_balance_and_development_exclusion():
    rows=approved_rows(); dev={r['patient_id'] for r in rows[:5]}
    split=freeze_patient_split(rows,20261006,dev)
    assert {k:len(v) for k,v in split.items()}=={'train':42,'validation':9,'test':9}
    ids={k:{r['patient_id'] for r in v} for k,v in split.items()}
    assert not (ids['train']&ids['validation'] or ids['train']&ids['test'] or ids['validation']&ids['test'])
    assert set.union(*ids.values())=={r['patient_id'] for r in rows}
    assert ids['test'].isdisjoint(dev)
    assert Counter(r['age_group'] for r in split['test'])=={'2-5':3,'6-11':3,'12-17':3}


def test_split_deterministic_and_independent_of_row_order():
    rows=approved_rows()
    assert freeze_patient_split(rows,42,set())==freeze_patient_split(list(reversed(rows)),42,set())


@pytest.mark.parametrize('change',['duplicate','unapproved','age'])
def test_split_rejects_unsafe_inputs(change):
    rows=approved_rows()
    if change=='duplicate':rows.append(rows[0])
    if change=='unapproved':rows[0]['review_status']='annotation_scope_review'
    if change=='age':rows[0]['age']=18
    with pytest.raises(ValueError):freeze_patient_split(rows,42,set())


def test_allocation_smaller_cohort_keeps_counts():
    assert sum(allocate(7,{'2-5':16,'6-11':16,'12-17':16}).values())==7
    split=freeze_patient_split(approved_rows()[:16]+approved_rows()[20:36]+approved_rows()[40:56],42,set())
    assert {k:len(v) for k,v in split.items()}=={'train':34,'validation':7,'test':7}


def test_split_rejects_duplicate_image_content_and_unresolved_related_ids():
    rows=approved_rows();rows[0]['image_voxel_SHA256']='same';rows[1]['image_voxel_SHA256']='same'
    with pytest.raises(ValueError,match='image duplicate'):freeze_patient_split(rows,42,set())
    rows=approved_rows();rows[0]['related_patient_ids']=[rows[1]['patient_id']]
    with pytest.raises(ValueError,match='related patient'):freeze_patient_split(rows,42,set())
