import os
import logging
import logging.config


log_config = {
    'disable_existing_loggers': False,
    'version': 1,
    'formatters': {
        'simple': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'maze_common': {
            'format': '%(asctime)s - %(name)s - %(filename)s - %(funcName)s - %(lineno)s - %(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'maze_common',
            'stream': 'ext://sys.stdout',
        },
        'maze_handler': {
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'formatter': 'maze_common',
            'filename': os.getenv('MAZELOG', '/tmp/maze.log'),
            'filename': os.path.join(os.path.dirname(__file__), 'logs/maze.log'),
            'encoding': 'utf-8',
            'when': 'W0',   # 每周一切割日志
        },
    },
    'loggers': {
        'console': {
            'level': 'INFO',
            'handlers': ['console'],
            'propagate': 0
        },
        'maze': {
            'level': 'INFO',
            'handlers': ['maze_handler', 'console'],
            'propagate': 0
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console']
    }
}


logging.config.dictConfig(log_config)


def get_logger(log_name):
    return logging.getLogger(log_name)


logger = get_logger("maze")
