import json
import pandas as pd
import matplotlib.pyplot as plt

files_with_acc = [
    'scalars_cbam.json', 'scalars_cbam_aspp.json', 'scalars_aspp.json',
    'scalars_mask_r101.json', 'scalars_mask_r50.json'
]

plt.figure(figsize=(10, 6))

for fn in files_with_acc:
    data = []
    with open(fn, 'r') as f:
        for line in f:
            try: data.append(json.loads(line))
            except: continue
    
    df = pd.DataFrame(data)
    # Nhóm theo Epoch và lấy trung bình
    epoch_acc = df.groupby('epoch')['acc'].mean().reset_index()
    
    # Chỉ vẽ 10 Epoch đầu tiên để xem điểm hội tụ ban đầu
    early_epochs = epoch_acc[epoch_acc['epoch'] <= 10]
    name = fn.replace('scalars_', '').replace('.json', '')
    
    plt.plot(early_epochs['epoch'], early_epochs['acc'], label=name, marker='o')

plt.title('Accuracy', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.xticks(range(1, 11))
plt.grid(True, ls='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('early_convergence_acc.png')
plt.show()