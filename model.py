from models import resnet, densenet, alexnet, vgg, mlp

class parse_opts():
    
    def __init__(self, model='resnet', model_hp=18):
        self.model = model                  # alexnet| resnet | densenet | vgg
        self.model_hp = model_hp         # hyperparameters of model like depth and input dim
        self.conv1_t_size = 7               # kernel size in t dim of conv1.
        self.conv1_t_stride = 2             # stride in t dim of conv1.
        self.no_max_pool = True             # if true, the max pooling after conv1 is removed.
        self.resnet_shortcut = 'B'          # shortcut type of resnet (A | B)
        self.resnet_widen_factor = 1.0      # the number of feature maps of resnet is multiplied by this value
        self.wide_resnet_k = 2              # wide resnet k
        self.resnext_cardinality = 32       # resNeXt cardinality
        self.n_classes = 1                  # number of classes (regression = 1)
        self.n_input_channels = 1           # number of input channels (distance grid = 1)

def generate_model(model='resnet', model_hp=18):
    opt = parse_opts(model, model_hp)
    return init_model(opt)

def init_model(opt):
    assert opt.model in ['resnet', 'densenet', 'alexnet', 'vgg', 'mlp']
       
    if opt.model == 'resnet':                                                     # model_depth should be: 10, 18, 34, 50, 101, 152, 200
        model = resnet.generate_model(model_depth=opt.model_hp,
                                      n_classes=opt.n_classes,
                                      n_input_channels=opt.n_input_channels,
                                      shortcut_type=opt.resnet_shortcut,
                                      conv1_t_size=opt.conv1_t_size,
                                      conv1_t_stride=opt.conv1_t_stride,
                                      no_max_pool=opt.no_max_pool,
                                      widen_factor=opt.resnet_widen_factor)

    elif opt.model == 'densenet':                                                 # model_depth should be: 121, 169, 201, 264
        model = densenet.generate_model(model_depth=opt.model_hp,
                                        num_classes=opt.n_classes,
                                        n_input_channels=opt.n_input_channels,
                                        conv1_t_size=opt.conv1_t_size,
                                        conv1_t_stride=opt.conv1_t_stride,
                                        no_max_pool=opt.no_max_pool)
    elif opt.model == 'alexnet':
        model = alexnet.AlexNet(input_dim=opt.model_hp)                        # here opt.model_depth means input_dim

    elif opt.model == 'vgg':
        model = vgg.generate_model(model_depth=opt.model_hp)                   # model_depth should be: 11, 13, 16, 19

    elif opt.model == 'mlp':
        model = mlp.MLP(n_hidden=opt.model_hp)

    return model
