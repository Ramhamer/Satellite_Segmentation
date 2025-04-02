from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology

base_model = GroundedSAM(ontology=CaptionOntology({"forest": "forest",}))

# label all images in a folder called `context_images`
base_model.label("./context_images", extension=".jpeg")
