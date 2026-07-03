import logging


def setup_logger(file_name):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('retriever_attack.log'),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(file_name)