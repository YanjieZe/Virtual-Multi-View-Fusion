try:
    from render.single_renderer import Renderer
except:
    from single_renderer import Renderer


class MultiRenderer(Renderer):
    def __init__(self):
        super(MultiRenderer, self).__init__()
        


