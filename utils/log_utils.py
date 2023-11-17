import logging

def cus_logger(args, name, m='a'):
    custom_logger = logging.getLogger(name)
    custom_logger.setLevel(logging.DEBUG)
    custom_logger.handlers.clear()

    fh = logging.FileHandler(f'./log/{args.log_name}.txt', mode=m)
    log_format = "[%(filename)s:%(lineno)d] %(message)s"
    fh.setFormatter(logging.Formatter(log_format))
    custom_logger.addHandler(fh)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(log_format))
    custom_logger.addHandler(console_handler)

    return custom_logger