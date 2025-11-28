import os
import json
import shutil
import pandas as pd
from pathlib import Path

class DataPreprocessor:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.train_move_object_text = []
        self.train_drop_object_text = []
        self.train_cover_object_text = []
        self.val_move_object_text = []
        self.val_drop_object_text = []
        self.val_cover_object_text = []
        self.labels_map = []

    def load_data(self):
        with open(self.base_dir / 'labels.json', 'r', encoding='utf-8') as f:
            labels_map = json.load(f)

        file_path = self.base_dir / 'train.json'
        if not file_path.exists():
            print('file not exists!')
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    if item.get('template') == 'Pushing [something] from left to right':
                        self.train_move_object_text.append(item)
                    elif item.get('template') == 'Dropping [something] onto [something]':
                        self.train_drop_object_text.append(item)
                    elif item.get('template') == 'Covering [something] with [something]':
                        self.train_cover_object_text.append(item)

        print(f'训练集中的move_object: {len(self.train_move_object_text)}')
        with open('train_move_object.json', 'w', encoding='utf-8') as f:
            json.dump(self.train_move_object_text, f, ensure_ascii=False, indent=2)
        print(f'训练集中的drop_object: {len(self.train_drop_object_text)}')
        with open('train_drop_object.json', 'w', encoding='utf-8') as f:
            json.dump(self.train_drop_object_text, f, ensure_ascii=False, indent=2)
        print(f'训练集中的cover_object: {len(self.train_cover_object_text)}')
        with open('train_cover_object.json', 'w', encoding='utf-8') as f:
            json.dump(self.train_cover_object_text, f, ensure_ascii=False, indent=2)


        file_path = self.base_dir / 'validation.json'
        if not file_path.exists():
            print('file not exists!')
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    if item.get('template') == 'Pushing [something] from left to right':
                        self.val_move_object_text.append(item)
                    elif item.get('template') == 'Dropping [something] onto [something]':
                        self.val_drop_object_text.append(item)
                    elif item.get('template') == 'Covering [something] with [something]':
                        self.val_cover_object_text.append(item)

        print(f'验证集中的move_object: {len(self.val_move_object_text)}')
        with open('val_move_object.json', 'w', encoding='utf-8') as f:
            json.dump(self.val_move_object_text, f, ensure_ascii=False, indent=2)
        print(f'验证集中的drop_object: {len(self.val_drop_object_text)}')
        with open('val_drop_object.json', 'w', encoding='utf-8') as f:
            json.dump(self.val_drop_object_text, f, ensure_ascii=False, indent=2)
        print(f'验证集中的cover_object: {len(self.val_cover_object_text)}')
        with open('val_cover_object.json', 'w', encoding='utf-8') as f:
            json.dump(self.val_cover_object_text, f, ensure_ascii=False, indent=2)

        self.train_move_object_text = pd.DataFrame(self.train_move_object_text)
        self.train_drop_object_text = pd.DataFrame(self.train_drop_object_text)
        self.train_cover_object_text = pd.DataFrame(self.train_cover_object_text)
        self.val_move_object_text = pd.DataFrame(self.val_move_object_text)
        self.val_drop_object_text = pd.DataFrame(self.val_drop_object_text)
        self.val_cover_object_text = pd.DataFrame(self.val_cover_object_text)

        self.train_text = pd.concat([self.train_move_object_text,
                                self.train_drop_object_text,
                                self.train_cover_object_text],
                               axis=0,
                               ignore_index=True)
        self.train_text.to_json('train_text.json', orient='records', lines=True)

        self.val_text = pd.concat([self.val_move_object_text,
                              self.val_drop_object_text,
                              self.val_cover_object_text],
                             axis=0,
                             ignore_index=True)
        self.val_text.to_json('val_text.json', orient='records', lines=True)

        return self.train_move_object_text, self.train_drop_object_text, self.train_cover_object_text, self.val_move_object_text, self.val_drop_object_text, self.val_cover_object_text, self.train_text, self.val_text, labels_map

    def copy_videos_by_id(self, current_dir):
        target_folder_move = os.path.join(current_dir, 'move')
        target_folder_drop = os.path.join(current_dir, 'drop')
        target_folder_cover = os.path.join(current_dir, 'cover')
        source_folder = os.path.join(current_dir, '20bn-something-something-v2')
        Path(target_folder_move).mkdir(parents=True, exist_ok=True)
        Path(target_folder_drop).mkdir(parents=True, exist_ok=True)
        Path(target_folder_cover).mkdir(parents=True, exist_ok=True)
        copied_count = {
            'move': 0,
            'drop': 0,
            'cover': 0,
        }
        for index, row in self.train_move_object_text.iterrows():
            if copied_count['move'] == 100:
                print(f"\nmove操作完成!")
                break
            id = row.get('id')
            source_path = os.path.join(source_folder, f"{id}.webm")
            target_path = os.path.join(target_folder_move, f"{id}.webm")
            try:
                shutil.copy2(source_path, target_path)
                copied_count['move'] += 1
                print(f"已复制: {id}.webm")
            except Exception as e:
                print(f"复制 {id}.webm 时出错: {e}")


        for index, row in self.train_drop_object_text.iterrows():
            if copied_count['drop'] == 100:
                print(f"\ndrop操作完成!")
                break
            id = row.get('id')
            source_path = os.path.join(source_folder, f"{id}.webm")
            target_path = os.path.join(target_folder_drop, f"{id}.webm")
            try:
                shutil.copy2(source_path, target_path)
                copied_count['drop'] += 1
                print(f"已复制: {id}.webm")
            except Exception as e:
                print(f"复制 {id}.webm 时出错: {e}")


        for index, row in self.train_cover_object_text.iterrows():
            if copied_count['cover'] == 100:
                print(f"\ncover操作完成!")
                break
            id = row.get('id')
            source_path = os.path.join(source_folder, f"{id}.webm")
            target_path = os.path.join(target_folder_cover, f"{id}.webm")
            try:
                shutil.copy2(source_path, target_path)
                copied_count['cover'] += 1
                print(f"已复制: {id}.webm")
            except Exception as e:
                print(f"复制 {id}.webm 时出错: {e}")



# 使用示例
if __name__ == "__main__":
    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, 'labels')
    preprocessor = DataPreprocessor(data_path)
    train_move_object_text, train_drop_object_text, train_cover_object_text, val_move_object_text, val_drop_object_text, val_cover_object_text, train_text, val_text, labels_map = preprocessor.load_data()
    preprocessor.copy_videos_by_id(current_dir)

