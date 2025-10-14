import os
import json
from tqdm import tqdm


base_path = os.path.dirname(os.path.abspath(__file__))
def process_longmemeval(data='longmemeval_s'):
    output_path = os.path.join(base_path, f'process_data/{data}.json')
    input_path = os.path.join(base_path, f'origin_data/{data}')
    in_data = json.load(open(input_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    alldata = []

    for entry in tqdm(in_data):
        answer_ids = []
        for cur_sess_id, sess_entry, ts in zip(entry['haystack_session_ids'], entry['haystack_sessions'], entry['haystack_dates']):
            for turn_id, turn in enumerate(sess_entry):
                if 'has_answer' in turn and turn['has_answer']==True:
                    temp_id = f"{cur_sess_id.replace('answer_','')}-turn_{turn_id}"
                    if temp_id not in answer_ids:
                        answer_ids.append(f"{cur_sess_id.replace('answer_','')}-turn_{turn_id}")

        sessions = []
        sessions_ids = []
        for sess_id, sess_entry in zip(entry['haystack_session_ids'], entry['haystack_sessions']):
            session = []
            sess_id = sess_id.replace('answer_','')
            for id, item in enumerate(sess_entry):
                session.append(f"[{item['role']}]: {item['content']}") 
            sessions.append(session)
            sessions_ids.append(sess_id)

        session_dict = dict(zip(sessions_ids, sessions))        
        dataform = {
            'qa': 
            [{            
                'question_id':entry['question_id'],
                "question": entry['question'],
                "question_type": entry['question_type'],
                "question_date":entry['question_date'],
                "answer": entry['answer'],
                "answer_ids": answer_ids
            }],
            'sessions_dates':entry['haystack_dates'],
            'sessions':session_dict
            }
        alldata.append(dataform)
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(alldata, f, ensure_ascii=False, indent=4)

def process_locomo():
    ## locomo10
    from tqdm import tqdm
    import json
    output_path = os.path.join(base_path, f'process_data/locomo.json')
    input_path = os.path.join(base_path, f'origin_data/locomo10.json')
    in_data = json.load(open(input_path))
    type_dict = {
        1:'multi-hop retrieval',
        2:'temporal reasoning',
        3:'open domain knowledge',
        4:'single-hop retrieval',
        5:'adversarial',
    }

    alldata = []

    for entry in tqdm(in_data):
        conversation_id = entry['sample_id'].replace("-","") 
        conversation = entry['conversation']
        sessions_ids = []
        sessions_dates = []
        sessions = []
        for j in range(100):
            if f'session_{j}' in conversation:
                sessions_ids.append(f"{conversation_id}_D{j}")
                sessions_dates.append(conversation[f'session_{j}_date_time'])
                session = []
                for dialog in conversation[f'session_{j}']:
                    if 'blip_caption' in dialog:
                        session.append(f"[{dialog['speaker']}]: {dialog['text']} The image Caption: {dialog['blip_caption']}")
                    else:
                        session.append(f"[{dialog['speaker']}]: {dialog['text']}")
                sessions.append(session)
        session_dict = dict(zip(sessions_ids, sessions)) 
        
        qa_list = []
        for qaitem in entry['qa']:
            if 'adversarial_answer' in qaitem:
                answer = qaitem['adversarial_answer']
            else:
                answer = qaitem['answer']
            answer_ids = []
            for answer_id in qaitem['evidence']:
                try:
                    turn_id = int(answer_id.split(':')[1])
                except:
                    continue
                id_parts = answer_id.split(":")
                session_part = id_parts[0]  # D1
                turn_part = int(id_parts[1])
                session_id = f"{conversation_id}_{session_part}"
                turn_id = f"turn_{turn_part - 1}"
                answer_ids.append(f"{session_id}-{turn_id}")
            qa_list.append(
                {
                "question": qaitem['question'],
                "question_type": type_dict[qaitem['category']],
                "answer": str(answer),
                "answer_ids":answer_ids,
            })
           
        dataform = {
            "qa": qa_list,
            'sessions_dates':sessions_dates,
            'sessions':session_dict
            }
        alldata.append(dataform)
        

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(alldata, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    process_longmemeval('longmemeval_s')
    process_locomo()
    # process_longmemeval('longmemeval_m')