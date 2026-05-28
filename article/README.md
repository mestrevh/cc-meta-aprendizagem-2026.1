# Descrição

Esse trabalho tem como objetivo a seleção de metafeatures visando a performance de sistemas de meta-aprendizado. A ideia é utilizar algoritmos de otimização para selecionar a melhor combinação de metafeture para a performance final.

Sugestões do professor:

- comparar o conjunto original de MFs com o conjunto final (reduzido);
- curva de performance ao longo do tempo de otimização;
- comparar conjunto reduzido com importância das features reportada por um modelo de Random Forest;
- discutir as possíveis causas da escolha de algumas features.

## Fase 1: Criação dos meta-datasets

1. Criação do metadataset (Exemplo 4)
2. Análise e discussão sobre os dados
3. Área de busca das metafeatures (atualmente tem mais de 1000 colunas)
---

## Fase 2: Classe de meta-modelos

1. Construção de uma classe para treinamento do meta-modelo com metodologia LeaveOneOut
2. Ideia é utilizar mais de 1 classificador para meta-modelo
3. Pré processamento nos meta-modelos
---

## Fase 3: Exploração dos algoritmos de seleção

1. Testar com algoritmos exatos (exemplo: programação linear - cplex) e algoritmos meta-heuristicos (exemplo: algoritmo genético)
2. 3 abordagens para ambos tipos de algoritmos
3. Utilizar uma abordagem híbrida: diminuir o espaço de busca e verificar a importância para melhorar 
---

## Fase 4: Resultados e discussão

1. Criar tabela com os resultados dos meta-modelos
2. Utilizar gráficos e falar sobre o impacto das meta-features
3. Concluir recomendando a melhor peformance sobre as características
---

### Referências

1. Section 4 - Metalearning Applications to Automated Machine Learning and Data Mining
2. Predicting relative performance of classifiers from samples.
3. An iterative process for building learning curves and predicting relative performance of classifiers.