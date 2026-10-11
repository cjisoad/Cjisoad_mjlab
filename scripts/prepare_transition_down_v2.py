"""Revalidate original teacher states against the approved V2 first frame."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import sys,json,hashlib,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from src.tasks.pedipulation_transition_t.motion import make_model
from src.tasks.pedipulation.constants import JOINT_NAMES

def prepare(source,reference,output):
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if sha(source)!='ff962e464cd928e44dea76f60363867f5361a45564ad7adbd6a43632d7193058':raise ValueError('Unreviewed source bank')
    if sha(reference)!='a806e849906846e66bf2421afad780132478a93f71c4b4f4a15300ae7b902646':raise ValueError('Unreviewed V2 reference')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with np.load(source,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files if k!='metadata_json'};metadata=json.loads(str(z['metadata_json']))
    with np.load(reference,allow_pickle=False) as z:refroot=z['qpos'][0,:7];reffoot=z['foot_pos_w'][0,1]
    model=make_model();d=mujoco.MjData(model);qadr=[model.joint(n).qposadr[0] for n in JOINT_NAMES];fr=model.body('FR_foot').id
    def heading(quat):
        left=Rotation.from_quat(quat[[1,2,3,0]]).apply([0,1,0]);f=np.cross(left,[0,0,1]);return np.arctan2(f[1],f[0])
    reference_heading=heading(refroot[3:7]);errors=np.zeros(len(arrays['root_state']))
    for i,root in enumerate(arrays['root_state']):
        d.qpos[:7]=root[:7];d.qpos[qadr]=arrays['joint_pos'][i];mujoco.mj_forward(model,d)
        rot=Rotation.from_euler('z',heading(root[3:7])-reference_heading)
        expected=rot.apply(reffoot);translation=root[:3]-rot.apply(refroot[:3]);translation[2]=0.
        errors[i]=np.linalg.norm(d.xpos[fr]-expected-translation)
    old=arrays['fr_error'].copy();arrays['fr_error']=errors.astype(old.dtype)
    arrays['actual_speed']=np.linalg.norm(arrays['root_state'][:,7:9],axis=1).astype(arrays['actual_speed'].dtype)
    metadata.update(motion_sha256=sha(reference),source_bank_sha256=sha(source),
        split_rule='train: trajectory_id%5!=0; validation:%10==5; final:%10==0',
        landing_version=2,FR_error_recomputed=True)
    arrays['metadata_json']=np.array(json.dumps(metadata,sort_keys=True))
    bank=output/'biped_entry_v2.npz';np.savez_compressed(bank,**arrays)
    target=output/'biped_to_tripod_v2.npz';target.write_bytes(Path(reference).read_bytes())
    ids=arrays['trajectory_id'];splits={name:len(np.unique(ids[mask])) for name,mask in
        [('train',ids%5!=0),('validation',ids%10==5),('final',ids%10==0)]}
    if splits!={'train':512,'validation':64,'final':64}:raise ValueError(splits)
    manifest=dict(version=2,motion_sha256=sha(target),bank_sha256=sha(bank),source_bank_sha256=sha(source),
        split_trajectories=splits,FR_error_recompute_max_delta_m=float(abs(errors-old).max()),
        max_observed_speed_m_s=float(arrays['actual_speed'].max()),max_observed_FR_error_m=float(errors.max()),
        state_history_fields_preserved=[k for k in arrays if k not in ('metadata_json','fr_error','actual_speed')],
        reference_provenance='src/assets/motions/go2/landing_v2/reference_provenance.md')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'outputs/biped_to_tripod_preparation_20261010/biped_entry_bank.npz')
    p.add_argument('--reference',type=Path,default=ROOT/'outputs/transition_down_landing_revision_20261011/biped_to_tripod_reference_v2.npz')
    p.add_argument('--output',type=Path,default=ROOT/'src/assets/motions/go2/landing_v2')
    a=p.parse_args();prepare(a.source,a.reference,a.output)
