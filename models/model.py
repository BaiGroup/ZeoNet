from models.cnn import resnet3d, densenet3d, alexnet3d, vgg3d, mvcnn_resnet
from models.gnn import cgcnn, m3gnet, megnet, mace_model
from models import dgcnn, vit

def get_model(config):
    if config.model.name == 'resnet3d':                                           # model_depth should be: 10, 18, 34, 50, 101, 152, 200
        model = resnet3d.generate_model(**config.model.model_params)

    elif config.model.name == 'densenet3d':                                       # model_depth should be: 121, 169, 201, 264
        model = densenet3d.generate_model(**config.model.model_params)
        
    elif config.model.name == 'vgg3d':
        model = vgg3d.generate_model(**config.model.model_params)                 # model_depth should be: 11, 13, 16, 19
    
    elif config.model.name == 'alexnet3d':
        model = alexnet3d.AlexNet(**config.model.model_params)          

    elif config.model.name == 'edgeconv':
        model = dgcnn.DGCNN(**config.model.model_params)
        
    elif config.model.name == 'vit3d':
        model = vit.VIT(**config.model.model_params)

    elif config.model.name == 'mvcnn_resnet18':
        model = mvcnn_resnet.MVCNN_resnet18(**config.model.model_params)

    elif config.model.name == 'pointnet':
        model = dgcnn.PointNet(**config.model.model_params)
        
    elif config.model.name == 'cgcnn':
        model = cgcnn.CrystalGraphConvNet(**config.model.model_params)
        
    elif config.model.name == 'megnet':
        model = megnet.MEGNet(**config.model.model_params)
        
    elif config.model.name == 'm3gnet':
        model = m3gnet.M3GNet(**config.model.model_params)

    elif config.model.name == 'mace':
        model = mace_model.ScaleShiftMACE(**config.model.model_params)

    else:
        raise ValueError(f"Model {config.model.name} not recognized.")
    return model
