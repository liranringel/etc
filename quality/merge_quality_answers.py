import torch

def main():
    num_parts = 5
    l = []

    for i in range(num_parts):
        l.append(torch.load(f'quality_answers_part{i}.pt'))
    
    all_answers = {'X': [], 'y': []}

    for i in range(num_parts):
        all_answers['X'] += l[i]['X']
        all_answers['y'] += l[i]['y']

    for i in range(len(all_answers['X'])):
        all_answers['X'][i]['all_scores'] = all_answers['X'][i]['all_scores'].cpu()
    
    torch.save(all_answers, 'quality_all_answers.pt')


if __name__ == '__main__':
    main()
