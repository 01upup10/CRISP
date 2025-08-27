import logging
import os

class Logger:
    '''
    logdir: tensorboard or tensorboardX save path
    log_filepath: logger save path
    '''
    def __init__(self, logdir, rank, type='tensorboard', debug=False, filename=None, summary=True, step=None, log_filepath=None):
        self.writer = None
        self.type = type
        self.rank = rank
        self.step = step

        self.summary = summary
        if summary:
            if type == 'tensorboardX':
                import tensorboardX
                self.writer = tensorboardX.SummaryWriter(logdir)
            elif type == 'tensorboard':
                from torch.utils import tensorboard
                self.writer = tensorboard.SummaryWriter(logdir)
            else:
                raise NotImplementedError
        else:
            self.type = 'None'
        # <<<<<logger>>>>>
        if log_filepath == None:
            log_filepath = os.path.join(logdir, 'logger.txt')
        elif os.path.isdir(log_filepath):
            log_filepath = os.path.join(log_filepath, 'logger.txt')
        self.log_filepath = log_filepath
        self.logger = self.create_logger(log_filepath=self.log_filepath)
        self.debug_flag = debug
        logging.basicConfig(filename=filename, level=logging.INFO, format=f'%(levelname)s:rank{rank}: %(message)s')

        if rank == 0:
            self.logger.info(f"[!] starting logging at directory {logdir}")
            # logging.info(f"[!] starting logging at directory {logdir}")
            if self.debug_flag:
                self.logger.info(f"[!] Entering DEBUG mode")
                # logging.info(f"[!] Entering DEBUG mode")

    def close(self):
        if self.writer is not None:
            self.writer.close()
        self.info("Closing the Logger.")

    def add_scalar(self, tag, scalar_value, step=None):
        if self.type == 'tensorboardX' or 'tensorboard':
            tag = self._transform_tag(tag)
            self.writer.add_scalar(tag, scalar_value, step)

    def add_image(self, tag, image, step=None):
        if self.type == 'tensorboardX' or 'tensorboard':
            tag = self._transform_tag(tag)
            self.writer.add_image(tag, image, step)

    def add_figure(self, tag, image, step=None):
        if self.type == 'tensorboardX' or 'tensorboard':
            tag = self._transform_tag(tag)
            self.writer.add_figure(tag, image, step)

    def add_table(self, tag, tbl, step=None):
        if self.type == 'tensorboardX' or 'tensorboard':
            tag = self._transform_tag(tag)
            tbl_str = "<table width=\"100%\"> "
            tbl_str += "<tr> \
                     <th>Term</th> \
                     <th>Value</th> \
                     </tr>"
            for k, v in tbl.items():
                tbl_str += "<tr> \
                           <td>%s</td> \
                           <td>%s</td> \
                           </tr>" % (k, v)

            tbl_str += "</table>"
            self.writer.add_text(tag, tbl_str, step)

    def add_graph(self, model, fake_input):
        if self.type == 'tensorboardX' or 'tensorboard':
            self.writer.add_graph(model, fake_input)
    
    def add_weights(self, model, global_step):
        '''
        model: model or model list
        '''
        if self.type == 'tensorboardX' or 'tensorboard':
            if not isinstance(model, list):
                for name, param in model.named_parameters():
                    self.writer.add_histogram(name, param, global_step)
            else:
                for i,m in enumerate(model):
                    for name, param in m.named_parameters():
                        self.writer.add_histogram(f"{name}/model_{i}", param, global_step)


    def print(self, msg):
        logging.info(msg)

    def info(self, msg):
        if self.rank == 0:
            self.logger.info(msg)
            # logging.info(msg)

    def debug(self, msg):
        if self.rank == 0 and self.debug_flag:
            logging.info(msg)

    def error(self, msg):
        logging.error(msg)

    def _transform_tag(self, tag):
        tag = tag + f"/{self.step}" if self.step is not None else tag
        return tag
    
    def add_results(self, results):
        if self.type == 'tensorboardX' or 'tensorboard':
            tag = self._transform_tag("Results")
            text = "<table width=\"100%\">"
            for k, res in results.items():
                text += f"<tr><td>{k}</td>" + " ".join([str(f'<td>{x}</td>') for x in res.values()]) + "</tr>"
            text += "</table>"
            self.writer.add_text(tag, text)

    def create_logger(self, log_filepath):

        _logger = logging.getLogger(__name__)
        _logger.setLevel(logging.INFO)  # 设置日志级别
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 创建文件处理器，并设置文件模式和路径
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setLevel(logging.INFO)

        # 创建格式化器并添加到处理器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # 将处理器添加到日志器
        # _logger.addHandler(console_handler)
        _logger.addHandler(file_handler)

        return _logger
    
if __name__ == "__main__":
    import os
    logdir_full = './output/{task_name}/experiment/{step}_{current_time}_lr{lr}'
    rank = 0
    step = 1
    log_path = os.path.join(logdir_full, 'logger.log')
    logger = Logger(logdir_full, rank=rank, debug=False, summary=True, step=step, log_filepath=log_path)
    logger.info("hello!")

