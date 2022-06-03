import time
from trainer.setting import Setting, Arguments
from trainer.models.processor import Processor


def main(args, logger) -> None:

    processor = Processor(args)
    config = processor.model_setting()

    logger.info('Model Setting Complete')

    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss, train_acc, train_f1 = processor.train()
        valid_loss, valid_acc, valid_f1 = processor.valid()

        end_time = time.time()
        epoch_mins, epoch_secs = processor.metric.cal_time(start_time, end_time)

        performance = {'tl': train_loss, 'vl': valid_loss,
                       'ta': train_acc, 'va': valid_acc,
                       'tf': train_f1, 'vf': valid_f1,
                       'ep': epoch, 'epm': epoch_mins, 'eps': epoch_secs}

        processor.metric.save_model(config, performance, processor.model_checker)

    """
    if args.test == 'True':
        logger.info("Start Test")

        hit_10, hit_args, mrr_k, _ = processor.test()
        print(f'\n\t== Hit@10: {hit_10:.4f} | Hit@{args.hit_k}: {hit_args:.4f} | MRR@{args.hit_k}: {mrr_k:.4f} ==\n')
    """

if __name__ == '__main__':
    args, logger = Setting().run()
    main(args, logger)
