import os

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#s.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
import tensorflow as tf
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from chat import preprocess_txt, load_chat_params, do_chat, check_in_list
from translator import translate, Encoder, Decoder
from tools import cur_time

# для чата
chat_params = load_chat_params()

# для переводчика
params_translator = {}
with open('./params_translator.pkl', 'rb') as f:
    params_translator = pickle.load(f)

encoder = Encoder(params_translator['vocab_inp_size'],
                  params_translator['embedding_dim'],
                  params_translator['units'],
                  params_translator['BATCH_SIZE'])

decoder = Decoder(params_translator['vocab_tar_size'],
                  params_translator['embedding_dim'],
                  params_translator['units'],
                  params_translator['BATCH_SIZE'])

params_translator['optimizer'] = tf.keras.optimizers.Adam()
params_translator['encoder'] = encoder
params_translator['decoder'] = decoder

checkpoint_dir = './training_nmt_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    optimizer=tf.keras.optimizers.Adam(),
    encoder=encoder,
    decoder=decoder,
)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def process(text, kwargs):
    sw = kwargs['sw']
    morpher = kwargs['morpher']
    exclude = kwargs['exclude']
    modelFT = kwargs['modelFT']
    ft_index = kwargs['ft_index']

    input_txt_all = preprocess_txt(line=text,
                                   sw=sw,
                                   morpher=morpher,
                                   exclude=exclude,
                                   skip_stop_word=False)

    # print(f'Слова в запросе: {input_txt_all}')

    # Говорим сколько время
    if check_in_list(input_txt_all, ['время', 'час']):
        return cur_time()

    # Делаем перевод
    if check_in_list(input_txt_all, ['перевести', 'перевод']):
        return translate(text, params_translator)

    # Простой чат
    return do_chat(text, kwargs)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf'Привет! Спроси меня сколько время, попроси ' + \
        rf'перевести фразу на английский язык или поболтай со мной',
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Help!")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    await update.message.reply_text(process(update.message.text, chat_params))


if __name__ == "__main__":
    bot_access = ''
    with open('./bot_access.txt', 'r') as f:
        for line in f:
            bot_access = line.replace('\n', '')
            break

    application = Application.builder().token(bot_access).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    application.run_polling(allowed_updates=Update.ALL_TYPES)