# from src.logger import logging 
# logging.debug("This is a debug message")
# logging.info("This is a info message")
# logging.warning("This is warning message")
# logging.error("'This is error message")

# from src.logger import logging
# from src.exception import MyException


# import sys

# try:
#     a=1+'Z'
# except Exception as e:
#     logging.info(e)
#     raise MyException(e,sys) from e


from src.pipline.training_pipeline import TrainPipeline  # or pipeline if that's correct

pipeline = TrainPipeline()
pipeline.run_pipeline()