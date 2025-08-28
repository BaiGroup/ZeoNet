from enum import Enum
from typing import Optional, Union, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
import yaml
from pathlib import Path
from trainer.grid_trainer import GridTrainer
from trainer.point_trainer import SurfacePointTrainer, CoordTrainer
from trainer.graph_trainer import CGCNNTrainer, MEGNETTrainer, M3GNETTrainer, MACETrainer
from trainer.image_trainer import ImageTrainer

def get_trainer(config):
    """Factory function to get the appropriate trainer"""
    if config.loader.representation == "grid":
        return GridTrainer(config)
    elif config.loader.representation == "surface_point":
        return SurfacePointTrainer(config)
    elif config.loader.representation == "coord":
        return CoordTrainer(config)
    elif config.loader.representation == "image":
        return ImageTrainer(config)
    elif config.loader.representation == "graph" and config.model.name == "cgcnn":
        return CGCNNTrainer(config)
    elif config.loader.representation == "graph" and config.model.name == "megnet":
        return MEGNETTrainer(config)
    elif config.loader.representation == "graph" and config.model.name == "m3gnet":
        return M3GNETTrainer(config)
    elif config.loader.representation == "graph" and config.model.name == "mace":
        return MACETrainer(config)
    else:
        raise ValueError(f"Unsupported representation type: {config.loader.representation}") 

class RepresentationType(str, Enum):
    IMAGE = "image"
    GRID = "grid"
    GRAPH = "graph"
    SURFACE_POINT = "surface_point"
    COORD = "coord"

class ModelType(str, Enum):
    """Supported model types"""
    RESNET3D = "resnet3d"
    DENSENET3D = "densenet3d"
    VGG3D = "vgg3d"
    ALEXNET3D = "alexnet3d"
    VIT3D = "vit3d"
    MVCNN_RESNET = "mvcnn_resnet18"
    POINTNET = "pointnet"
    EDGECONV = "edgeconv"
    CGCNN = "cgcnn"
    MEGNET = "megnet"
    M3GNET = "m3gnet"
    MACE = "mace"

class FlexibleBaseModel(BaseModel):
    """Base model that allows extra fields from config file"""
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

class LoaderConfig(FlexibleBaseModel):
    """Data configuration"""
    representation: RepresentationType = Field(..., description="Type of representation")
    data_path: str = Field(..., description="Path to data")
    batch_size: int = Field(16, description="Batch size")

class ModelConfig(FlexibleBaseModel):
    """Model configuration"""
    name: ModelType = Field(..., description="Type of model to use")
    model_params: Optional[Dict[str, Any]] = Field(None, description="Model-specific parameters")

class TrainingConfig(FlexibleBaseModel):
    """Training configuration"""
    max_epochs: int = Field(30, description="Number of epochs")
    learning_rate: float = Field(0.001, description="Learning rate")
    weight_decay: float = Field(0.0, description="Weight decay")
    use_gpu: bool = Field(True, description="Whether to use GPU")
    save_dir: str = Field("", description="Directory to save checkpoints")
    resume_path: str = Field("", description="Path to resume training from")

class Config(FlexibleBaseModel):
    """Main configuration class"""
    loader: LoaderConfig
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'Config':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return super().dict(*args, **kwargs)

    def save_yaml(self, yaml_path: Union[str, Path]):
        """Save configuration to YAML file"""
        config_dict = self.model_dump()
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

def load_config(config_path: Union[str, Path]) -> Config:
    """Load configuration from YAML file"""
    return Config.from_yaml(config_path)
