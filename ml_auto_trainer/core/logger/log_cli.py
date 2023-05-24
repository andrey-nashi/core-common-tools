class CliLogger:

    def __init__(self):
        pass

    def log_info(self, message):
        print("[INFO]:", message)

    def log_error(self, message):
        print("[ERROR]:", message)