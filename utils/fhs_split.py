

def main(): # TODO
    with open(manifest_dir / f'all.tsv', 'r') as f_tsv:
        lines = f_tsv.read().strip().split('\n')
        root = lines.pop(0)
        data_dict = defaultdict(list)
        for l in tqdm(lines):
            wav_path = os.path.join(root, l.strip().split('\t')[0])
            id_date = Path(wav_path).parent.name
            start, end = Path(wav_path).stem.split('_')[-2:]
            start = int(start)
            end = int(end)
            if id_date in id_dates:
                id_date_idx = id_dates.index(id_date)
                has_review = has_reviews[id_date_idx]
                label = 0
                if has_review:
                    is_norm = is_norms[id_date_idx]
                    label = int(is_norm == 0)
                data_dict[id_date].append([start, end, [wav_path], label])

                if count[label] < 10:
                    if not id_date in self.valid_id_dates:
                        count[label] += 1
                        self.valid_id_dates.append(id_date)
                elif split == 'train': 
                    self.train_id_dates.append(id_date)
                elif count.sum() >= 20:
                    break


