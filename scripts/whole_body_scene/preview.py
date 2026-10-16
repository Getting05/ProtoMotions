"""Scientific inspection plots/video of reference replay and reconstructed solids."""
import argparse
import shutil
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation
from .geometry import CollisionModel
from .extract_motion import MotionArchive


def draw_boxes(ax,scene):
    vertices=np.array([[x,y,z] for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]])
    faces=[[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
    for entity in scene["objects"]:
        for box in entity["primitives"]:
            v=(vertices*np.array(box["size"]))@Rotation.from_quat(box["rotation"]).as_matrix().T+box["translation"]
            color="#94a3b8" if box["role"]=="ground" else "#e0a35a" if box["role"]=="inferred_support" else "#4ca49c"
            ax.add_collection3d(Poly3DCollection(v[faces],facecolors=color,edgecolors=color,alpha=.25,linewidths=.5))


def prepare(record,root,archive,model):
    motion=archive.get(record.get("source_motion_local_id",record["motion_local_id"]))
    motion.pos+=np.array(record.get("scene_origin_offset",[0,0,0]))
    points,_,_=model.world_samples(motion)
    scene=json.loads((root/record["scene_file"]).read_text())
    quality=json.loads((root/record["quality_file"]).read_text())
    labels=np.load(root/record["contact_file"],allow_pickle=False)
    return motion,points,scene,quality,labels


def setup_axis(ax,points):
    lo=points.min(axis=(0,1));hi=points.max(axis=(0,1))
    center=(lo+hi)/2;radius=max(.6,float((hi-lo).max())/2+.15)
    ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius)
    ax.set_zlim(max(-.06,center[2]-radius),center[2]+radius)
    ax.set_box_aspect((1,1,1));ax.view_init(20,-55)
    ax.set_xlabel("x (m)",fontsize=8);ax.set_ylabel("y (m)",fontsize=8);ax.set_zlabel("z (m)",fontsize=8)
    ax.tick_params(labelsize=7)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True);p.add_argument("--output",required=True)
    p.add_argument("--model",default="protomotions/data/assets/astro_p2/mjcf/astro_p2_protomotions.xml")
    p.add_argument("--video-id",type=int)
    p.add_argument("--audit-ids",help="Comma-separated source IDs for collision-height diagnostics")
    args=p.parse_args();manifest=Path(args.manifest);root=manifest.parent
    records=[json.loads(x) for x in manifest.read_text().splitlines() if x]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    model=CollisionModel(args.model,12)
    archive=MotionArchive(records[0].get("source_motion_shard",records[0]["motion_shard"]))
    fig=plt.figure(figsize=(16,4.7*int(np.ceil(len(records)/4))))
    coverage=[];summaries=[]
    for i,r in enumerate(records):
        motion,points,scene,q,labels=prepare(r,root,archive,model)
        frame=int(np.argmax((labels["link_contact_state"]==1).sum(axis=1)))
        ax=fig.add_subplot(int(np.ceil(len(records)/4)),4,i+1,projection="3d")
        draw_boxes(ax,scene)
        ax.scatter(*points[frame,::3].T,s=2,c="#436bad",alpha=.6)
        ax.plot(*motion.pos[:,0].T,color="#22314e",linewidth=1)
        setup_axis(ax,points)
        name=Path(r["motion_key"]).stem
        ax.set_title(f"ID {r['motion_local_id']} | {r['status']}\n{name[:38]}",fontsize=8)
        coverage.append((labels["link_contact_state"]==1).mean(0))
        summaries.append({"id":r["motion_local_id"],"key":r["motion_key"],"status":r["status"],
                          "offset":r.get("scene_origin_offset",[0,0,0]),"objects":len(scene["objects"]),
                          "contacts":q["contacts"],"support":{k:v for k,v in q["support"].items() if k!="samples"},
                          "reasons":r["reasons"],"runtime_seconds":q["runtime_seconds"]})
        labels.close()
    fig.suptitle("BONES-SEED / Astro P2: reference replay with reconstructed collision geometry\nGeometry hypotheses; this figure is not a closed-loop tracking result",fontsize=13)
    fig.subplots_adjust(left=.025,right=.975,bottom=.04,top=.88,hspace=.30,wspace=.08)
    fig.savefig(out/"scene_gallery.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,9))
    matrix=np.array(coverage).T*100
    im=ax.imshow(matrix,aspect="auto",vmin=0,vmax=100,cmap="viridis")
    ax.set_yticks(range(len(model.body_names)),model.body_names,fontsize=8)
    ax.set_xticks(range(len(records)),[str(r["motion_local_id"]) for r in records])
    ax.set_xlabel("Source motion local ID");ax.set_title("Explained contact frames (% of clip)\nDetector coverage only; not human-labeled precision")
    fig.colorbar(im,ax=ax,label="% frames");fig.tight_layout();fig.savefig(out/"body_contact_coverage.png",dpi=160);plt.close(fig)
    (out/"inspection_summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
    if args.audit_ids:
        audit_ids=list(map(int,args.audit_ids.split(",")))
        fig,axes=plt.subplots(len(audit_ids),1,figsize=(12,3.3*len(audit_ids)),squeeze=False)
        audit=[]
        for ax,mid in zip(axes[:,0],audit_ids):
            r=next(r for r in records if r["motion_local_id"]==mid)
            motion,points,scene,q,labels=prepare(r,root,archive,model)
            item={"source_motion_id":mid,"key":motion.key,"scene_origin_offset":r["scene_origin_offset"],
                  "metric":"sampled_collision_surface_lower_height_m","bodies":{}}
            for name in ("left_ankle_roll_link","right_ankle_roll_link","pelvis","waist_roll_link"):
                bid=model.body_names.index(name);mask=model.surface.body==bid
                z=points[:,mask,2].min(axis=1)
                ax.plot(np.arange(len(z))*motion.dt,z,label=name,linewidth=1.6)
                item["bodies"][name]={"min_m":float(z.min()),"max_m":float(z.max()),
                                      "median_m":float(np.median(z)),"trajectory_m":z.tolist()}
            ax.set_title(f"Source ID {mid}: {Path(motion.key).stem}",fontsize=10)
            ax.set_xlabel("Time (s)");ax.set_ylabel("Lower height (m)")
            ax.grid(alpha=.2);ax.legend(loc="upper right",fontsize=8,ncol=2)
            audit.append(item);labels.close()
        fig.suptitle("Source motion audit: one constant world translation per clip; no framewise grounding",fontsize=12)
        fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/"source_height_audit.png",dpi=160);plt.close(fig)
        (out/"source_height_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    if args.video_id is not None:
        if not shutil.which("ffmpeg"):
            import imageio_ffmpeg
            plt.rcParams["animation.ffmpeg_path"]=imageio_ffmpeg.get_ffmpeg_exe()
        r=next(r for r in records if r["motion_local_id"]==args.video_id)
        motion,points,scene,q,labels=prepare(r,root,archive,model)
        fig=plt.figure(figsize=(8,7));ax=fig.add_subplot(111,projection="3d")
        draw_boxes(ax,scene);setup_axis(ax,points)
        cloud=ax.scatter(*points[0,::2].T,s=5,c="#4064af",alpha=.65)
        title=ax.set_title("")
        def update(t):
            cloud._offsets3d=points[t,::2].T
            title.set_text(f"P2 / source ID {args.video_id} / {r['status']}\nReference replay, t={t*motion.dt:.2f}s; not closed-loop")
            return cloud,title
        animation=FuncAnimation(fig,update,frames=len(points),interval=motion.dt*1000,blit=False)
        animation.save(out/f"reference_scene_{args.video_id}.mp4",writer=FFMpegWriter(fps=1/motion.dt,bitrate=1800),dpi=110)
        plt.close(fig);labels.close()


if __name__=="__main__":main()
