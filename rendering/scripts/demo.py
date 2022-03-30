import yaml,sys 
sys.path += ['./', '../']
from src import FRender
import src.FASTRENDERING as FASTRENDERING

with open(sys.argv[1],"r") as f:
    cfg = yaml.load(f.read())

FR = FRender(**cfg)

FASTRENDERING.render_to_screen(FR, FR.W, FR.H, FR.num_group)