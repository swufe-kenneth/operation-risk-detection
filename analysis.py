# encoding: utf-8
import numpy as np


def gen_tabular_embeddings(original_datasets, embedding_rets, focal_entity):
    if focal_entity == 'staff':
        node_type = 'created_staff_nodes'
        data_index = 0
    if focal_entity == 'pc':
        node_type = 'created_pc_nodes'
        data_index = 2

    entities = {entity for date in original_datasets for entity in getattr(original_datasets[date], focal_entity)}

    tabular_embeddings = []
    for entity in entities:
        non_emtpy_entities = []
        dates = []
        embeddings = []
        for date in original_datasets:
            index = getattr(original_datasets[date], node_type).get(entity)
            if index is None:
                continue
            index = index[0]
            embedding = embedding_rets[date][data_index][index, :].detach().numpy()
            embeddings.append(embedding)

            dates.append(date)
            non_emtpy_entities.append(entity)

        non_emtpy_entities = np.array(non_emtpy_entities).reshape(-1, 1)
        dates = np.array(dates).reshape(-1, 1)
        embeddings = np.vstack(embeddings)
        tabular_embeddings.append(np.hstack((non_emtpy_entities, dates, embeddings)))

    return np.vstack(tabular_embeddings)


def ext_desired_entities(tabular_embeddings, alpha):
    desired_entities = []
    for entity in np.unique(tabular_embeddings[:, 0].ravel()):
        index = tabular_embeddings[:, 0] == entity
        if index.sum() < alpha: continue

        desired_entities.append(entity)
    return desired_entities
