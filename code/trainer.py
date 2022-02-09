import os

from utils import get_logger
from transformers import AdamW, get_linear_schedule_with_warmup
from fastprogress.fastprogress import master_bar, progress_bar

from data_utils import get_dataloader
import torch
from torch.nn import CrossEntropyLoss, BCELoss
from sklearn.metrics import accuracy_score


def train(args, model):
    ### Set Loggers ###
    logger = get_logger(args)
    device = model.model.device.type
    tokenizer = model.tokenizer
    
    train_dataloader = get_dataloader(args, tokenizer, data_type='train')

    best_acc = 0
    global_step = 1
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader.dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Train batch size per GPU = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(t_total * args.warmup_proportion), num_training_steps=t_total
    )
    criterion = BCELoss()

    steps_trained_in_current_epoch=0
    mb = master_bar(range(int(args.num_train_epochs)))

    for epoch in mb:
        epoch_iterator = progress_bar(train_dataloader, parent=mb)
        train_loss = 0
        model.train()
        for step, batch in enumerate(epoch_iterator, 1):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            input_ids, attention_mask, token_type_ids = inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']

            outputs = model(input_ids, attention_mask, token_type_ids)
            outputs = outputs.squeeze(-1)
            loss = criterion(outputs, labels)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            train_loss += loss.item()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step() # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.evaluate_during_training:
                        eval_loss, eval_acc = evaluate(args, model, criterion)
                        progress = global_step/t_total
                        
                        logger.info(f"epoch: {epoch}, step: {global_step}, eval_loss: {eval_loss.item():.4f}, acc: {eval_acc:.4f}, progress: {progress:.2f}")

             # Save model checkpoint
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    if eval_acc > best_acc:
                        best_acc = eval_acc
                        output_dir = os.path.join(args.output_dir, "best_checkpoint")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Take care of distributed/parallel training
                        torch.save(model, os.path.join(output_dir, 'intent_class_model.pt'))
                        tokenizer.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        
                        logger.info("Saving model checkpoint to %s", output_dir)
        mb.write("Epoch {} done".format(epoch+1))

    return global_step, train_loss/global_step

def evaluate(args, model, criterion):
    device = model.model.device.type
    tokenizer = model.tokenizer
    valid_dataloader = get_dataloader(args, tokenizer, data_type='valid')

    epoch_iterator = progress_bar(valid_dataloader)
    eval_loss = 0
    total_acc = 0
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator, 1):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            input_ids, attention_mask, token_type_ids = inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']

            outputs = model(input_ids, attention_mask, token_type_ids)
            outputs = outputs.squeeze(-1)

            loss = criterion(outputs, labels)
            eval_loss += loss.item()

            preds = (outputs > 0.5)

            flatten_preds = torch.flatten(preds).tolist()
            flatten_labels = torch.flatten(labels).tolist()

            acc_score = accuracy_score(flatten_preds, flatten_labels)
            total_acc += acc_score

    return eval_loss/step, total_acc/step