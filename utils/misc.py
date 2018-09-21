import os

def process_emoji_dir(emo_dir):
    assert os.path.isdir(emo_dir)
    emo_list = []
    for emo_name in os.listdir(emo_dir):
        if emo_name.endswith(('.png','.jpg','.jpeg')):
            assert emo_name not in emo_list,'Repeated names in emoji dir!'
            emo_list.append(emo_name)
    return emo_list