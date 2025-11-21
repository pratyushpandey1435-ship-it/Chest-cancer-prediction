from src.cnnClassifier import logger
from src.cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from src.cnnClassifier.pipeline.stage_03_model_trainer import ModelTrainerPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<")
    DataIngestion = DataIngestionTrainingPipeline()
    DataIngestion.main()
    logger.info(f">>>>>>> stage  {STAGE_NAME} completed <<<<<<<<\n\nx=============x")
except Exception as e:
    logger.exception(e)
    raise(e)


STAGE_NAME = "Prepare Base Model"

try:
    logger.info(f"************************")
    logger.info(f">>>>>>>>> stage {STAGE_NAME}  started <<<<<<<<<<<")
    PrepareBaseModel = PrepareBaseModelTrainingPipeline()
    PrepareBaseModel.main()
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx===============x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Training"

try:
    logger.info(f"************************")
    logger.info(f">>>>>>>>> stage {STAGE_NAME}  started <<<<<<<<<<<")
    Training = ModelTrainerPipeline()
    Training.main()
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx===============x")
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Evaluation"

try:
    logger.info(f"************************")
    logger.info(f">>>>>>>>> stage {STAGE_NAME}  started <<<<<<<<<<<")
    obj = ModelEvaluationPipeline()
    obj.main()
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx===============x")
except Exception as e:
    logger.exception(e)
    raise e