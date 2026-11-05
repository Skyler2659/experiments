"""CodeWorld-only perception wording; native manipulation lessons are retained."""

import re


def codeworld_knowledge(text):
    text = re.sub(r'## Perception\n.*?(?=\n## |\Z)', '''## Perception
- Read the current RGB images and CodeWorld. The three calibrated cameras are
  cam_head, cam_left_wrist and cam_right_wrist. Use the images to establish object
  identity, visibility, contact and change; geometry is an estimate.
- On slip, release, tilt or a collision, correct every visibly affected object
  in the existing post-action review. Check the persistent pose_refresh list.
  A changed world revision alone does not mean those objects were re-localised.
- Use current calibrated image correspondences with CodeWorld local geometry to
  fit a changed pose, or submit a visually estimated transform. Check fit residuals
  against the views; a low residual alone does not establish object identity.
- Occluded or ambiguous objects remain unresolved with their last estimate visible.
  Do not use an old carried pose as a measured landing position. Continue with the
  information available; no extra motion or wait is required just to clear a review.
- robo.image(camera) returns the recorded RGB for your own analysis.
''', text, flags=re.S)
    text = text.replace('(`locate`)', '(current RGB and CodeWorld)')
    text = text.replace('moved with the arm after a lift (wrist camera, or `locate` it at its old\n'
                        '  place and see it gone).  A held object is hard to `locate` because the\n'
                        '  gripper covers it.',
                        'moved with the arm after a lift using the wrist and head images.\n'
                        '  The gripper can occlude a held object.')
    text = text.replace('`locate` the object;', 'read and visually check the object in CodeWorld;')
    text = text.replace('`locate` it again from there (see Perception); turn to `yaw=grasp_yaw` and',
                        'correct its pose from the images; choose yaw across its narrow side and')
    text = text.replace('Before finishing, `locate`\n  each object again,',
                        'Before finishing, visually check each named object and its world state,')
    # Other RoboDojo knowledge variants use slightly different prose. Remove
    # service-specific paragraphs, without changing unrelated motion lessons.
    paragraphs = re.split(r'(?=^- )', text, flags=re.M)
    return ''.join(p for p in paragraphs if not re.search(r'\b(locate|back_project|segment)\b', p))
