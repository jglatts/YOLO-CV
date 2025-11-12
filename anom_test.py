'''
        Test Script to train and test a model on MVTecAD dataset
        NOTE:
            This script must be run in admin terminal to avoid permission issues
'''
from anomalib.data import MVTecAD
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore


def createEngineSimple(dataset, category_name):
    datamodule = Folder(
        root=dataset,
        normal_dir="train/good",      # where your normal samples are
        abnormal_dir="test/bad",   # where your defects are
        normal_split_ratio=0.8,       # auto split
        image_size=(256, 256),
    )

    model = Patchcore(num_neighbors=6)
    engine = Engine(max_epochs=1)
    engine.fit(datamodule=datamodule, model=model)
    engine.test(datamodule=datamodule, model=model)


# Note datasets must follow MVTecAD structure
def createEngineMVTecAD(dataset, category_name):
    # Create dataset
    datamodule = MVTecAD(
        root=dataset,
        category=category_name,
        train_batch_size=8,
        eval_batch_size=8,
    )

    # Initialize model and engine
    model = Patchcore(num_neighbors=6)
    engine = Engine(max_epochs=1)

    # Train and Test
    engine.fit(datamodule=datamodule, model=model)
    test_results = engine.test(datamodule=datamodule, model=model)



if __name__ == "__main__":
    #createEngineMVTecAD("./datasets/MVTecAD", "transistor")
    createEngineSimple("./datasets/Z-Axis", "zfill")
