if __name__ == "__main__":
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    # 1. Create dataset
    datamodule = MVTecAD(
        root="./datasets/MVTecAD",
        category="bottle",
        train_batch_size=32,
        eval_batch_size=32,
    )

    # 2. Initialize model and engine
    model = Patchcore(num_neighbors=6)
    engine = Engine(max_epochs=1)

    # 3. Train
    engine.fit(datamodule=datamodule, model=model)

    # 4. Test
    test_results = engine.test(datamodule=datamodule, model=model)
