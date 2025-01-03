import numpy as np
from keras import ops, optimizers, saving


@saving.register_keras_serializable(name="WarmUpCosine")
class WarmUpCosine(optimizers.schedules.LearningRateSchedule):
    """
    WarmUpCosine is a learning rate schedule that uses a warmup period followed by a cosine decay.
    """

    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super().__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = ops.convert_to_tensor(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")

        cos_annealed_lr = ops.cos(
            self.pi
            * (ops.cast(step, "float32") - self.warmup_steps)
            / float(self.total_steps - self.warmup_steps)
        )
        learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr)

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * ops.cast(step, "float32") + self.warmup_learning_rate
            learning_rate = ops.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return ops.where(
            step > self.total_steps,
            0.0,
            learning_rate,
        )

    def get_config(self):
        return {
            "learning_rate_base": self.learning_rate_base,
            "total_steps": self.total_steps,
            "warmup_learning_rate": self.warmup_learning_rate,
            "warmup_steps": self.warmup_steps,
        }


def get_lr_schedule(
    train_ds,
    args,
    downstream=False,
):
    total_steps = int(train_ds.cardinality().numpy() * args.epochs)
    warmup_steps = int(total_steps * args.warmup_epoch_percentage)

    # if downstream:
    #    learning_rate_base = args.learning_rate_base
    # else:
    #    learning_rate_base = (
    #        (
    #            np.sqrt(
    #                args.hs_decoder_mask_proportion * args.rgb_decoder_mask_proportion
    #            )
    #        )
    #        * args.learning_rate_base
    #        / (np.sqrt(args.hs_mask_proportion * args.rgb_mask_proportion))
    #    )

    #

    scheduled_lrs = WarmUpCosine(
        learning_rate_base=args.learning_rate_base,
        total_steps=total_steps,
        warmup_learning_rate=args.warmup_learning_rate,
        warmup_steps=warmup_steps,
    )

    # lrs = [scheduled_lrs(step) for step in range(total_steps)]

    # fig = plt.figure(figsize=(10, 7))
    # plt.plot(lrs)
    # plt.xlabel("Step", fontsize=14)
    # plt.ylabel("LR", fontsize=14)
    # plt.title("Learning Rate Schedule", fontsize=14)
    # plt.show()

    # wandb.log({"lr_schedule": plt})

    # plt.close(fig)

    # if not downstream:
    #    plt.savefig(f"plots/lr_schedule.png")
    # else:
    #    plt.savefig(f"plots/lr_schedule_downstream.png")

    return scheduled_lrs
