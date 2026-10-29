"""Isaac render-only estimated P2. Never creates physics/collision objects."""
import io
from PIL import Image
from protomotions.utils.reference_video import ReferenceRobot
from protomotions.utils.policy_video import FollowCamera as PolicyCamera


class EstimatedRobot(ReferenceRobot):
    def __init__(self,env):
        super().__init__(env)
        from pxr import Gf,UsdShade
        # The original blue reference is not spawned by this script. Reuse its
        # visual-only geometry, recolor red, drive it exclusively with estimates.
        shader=UsdShade.Shader.Get(self.stage,self.path+'/BlueReference/Surface')
        shader.GetInput('diffuseColor').Set(Gf.Vec3f(1.0,0.12,0.12))
        self.set_visible(False)


def jpeg(frame):
    output=io.BytesIO()
    Image.fromarray(frame).save(output,format='JPEG',quality=88)
    return output.getvalue()


class FollowCamera(PolicyCamera):
    """Closer following view: keep a P2-sized robot legible next to charts."""
    def _follow(self):
        import numpy as np
        from pxr import Gf
        root=self.simulator._get_simulator_root_state(0).root_pos.detach().cpu().numpy().reshape(-1,3)[0]
        eye=root+np.array([1.6,-2.2,.9])
        target=root+np.array([0.,0.,.05])
        view=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye.tolist()),Gf.Vec3d(*target.tolist()),Gf.Vec3d(0,0,1))
        self.transform.Set(view.GetInverse())
