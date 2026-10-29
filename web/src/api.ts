export type Point = [number,number,number];
export type Contour = {id:string,label:number,axis:number,slice:number,points:Point[],closed:boolean,applied:boolean,replace_label_slice?:boolean};
export type Operation = {kind:'contour'|'brush',contour_id?:string,indices?:number[],values?:number[]};
export type ReviewFlag = {id:string,axis:number,slice:number,label:number,kind:'boundary'|'missing_structure'|'artifact'|'uncertain'};
export type AnnotationDocument = {contours:Contour[],operations:Operation[],base_prediction_id?:string|null,flags?:ReviewFlag[]};
export type SliceScope = {axis:2,index:number,complete_volume:false};
export type Prediction = {id:string,number:number,source:string,approved:false,provenance:{variant:'v1'|'v2'|'slice_draft'|'anatomy_slice_draft',exploratory_only:boolean,absent_classes?:string[],scope?:SliceScope,method?:string,warning?:string,review_required_classes?:string[],notebook_version_id?:string,notebook_document_sha256?:string,notebook_export_sha256?:string,anatomy_bundle_sha256?:string,observer_sources?:Array<'human'|'local_model'|'cloud_model'>,medical_correctness_verified?:false}};
export type Label = {id:number,code:string,name:string,color:string};
export type Project = {id:string,patient_id:string,study_id:string,status:string,revision:number,updated_at:string,
  version_id:string|null,version_number:number,progress_classes:number,synthetic:boolean,ct_sha256:string,label_schema:string,
  geometry:{shape:Point,affine_ras:number[][],origin_lps:Point,spacing:Point,direction_lps:number[]},labels:Label[],document:AnnotationDocument,predictions:Prediction[],annotation_scope?:SliceScope|null};
export function sliceScope(project:Project,comparison:string):SliceScope|undefined {
  if(comparison==='draft')return project.annotation_scope||project.predictions?.find(p=>p.id===project.document.base_prediction_id)?.provenance.scope;
  return project.predictions?.find(p=>p.id===comparison)?.provenance.scope;
}
export type User = {username:string,role:string,csrf:string};
export type AnatomyConfidence = 'low'|'medium'|'high';
export type AnatomyLocator = {view:'AX'|'COR'|'SAG',anchor_ijk:Point,focal_point_lps_mm:Point,normal_lps:Point,
  native_axis?:number,native_index?:number,window_width:number,window_level:number,overlay_visible:false};
export type AnatomyObservation = {id:string,kind:'observation'|'hypothesis',structure:number,statement:string,features:string[],
  confidence:AnatomyConfidence,uncertainty_reason:string,evidence_refs:string[],source:'human'|'local_model'|'cloud_model',locator:AnatomyLocator};
export type AnatomyRelation = {from_id:string,to_id:string,kind:'continuation'|'branch'|'unresolved',gap_reason:string};
export type AnatomyTrack = {id:string,structure:number,observation_ids:string[],relations:AnatomyRelation[]};
export type AnatomyDecision = {label:number,state:'unassessed'|'delineated'|'uncertain'|'not_visible'|'not_assessable',
  supporting_observation_ids:string[],counterevidence_ids:string[],track_ids:string[],boundary_rule:string,rationale:string,
  unresolved_questions:string[],confidence:AnatomyConfidence};
export type AnatomyDocument = {schema_version:1,target:{axis:2,index:number}|null,observations:AnatomyObservation[],tracks:AnatomyTrack[],decisions:AnatomyDecision[]};
export type AnatomyReviewVersion = {version_id:string|null,number:number,ct_sha256:string,document:AnatomyDocument,readiness:{ready:boolean,issues:string[]}};
let csrf = '';
export function session(user:User|null) { csrf = user?.csrf || ''; }
export async function api<T>(path:string, options:RequestInit={}):Promise<T> {
  const response = await fetch('/api'+path,{...options,credentials:'same-origin',headers:{
    ...(options.body ? {'Content-Type':'application/json'} : {}),...(csrf ? {'X-CSRF-Token':csrf}:{}),...options.headers}});
  if(!response.ok) { const data=await response.json().catch(()=>({detail:response.statusText}));
    throw new Error(`${response.status}: ${typeof data.detail==='string'?data.detail:JSON.stringify(data.detail)}`); }
  return (response.headers.get('content-type')?.includes('application/json') ? response.json() : response.arrayBuffer()) as Promise<T>;
}
export const post = <T>(path:string,body:unknown={})=>api<T>(path,{method:'POST',body:JSON.stringify(body)});
export const copy = <T>(value:T):T=>structuredClone(value);
