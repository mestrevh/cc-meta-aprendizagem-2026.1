# Descrição

Esse trabalho tem como objetivo a seleção de metafeatures visando a performance de sistemas de meta-aprendizado. A ideia é utilizar algoritmos de otimização para selecionar a melhor combinação de metafeture para a performance final.

Sugestões do professor:

- comparar o conjunto original de MFs com o conjunto final (reduzido);
- curva de performance ao longo do tempo de otimização;
- comparar conjunto reduzido com importância das features reportada por um modelo de Random Forest;
- discutir as possíveis causas da escolha de algumas features.

## Fase 1
---

Testar com algoritmos exatos (exemplo: programação linear - cplex) e meta-heuristicos (exemplo: algoritmo genetico)

### Referências

1. Meta-Learning: A Survey
2. Report on the experiments with feature selection in meta-level learning;
3. 
4.  


### Tarefas para fazer

0. Criação do dataset com openml
1. Fazer um meta-modelo para avaliar as meta-features selecionadas
2. Criação da população inicial com tipos de algoritmos diferentes como na atividade 4
3. Verificar se é possivel utilizar algoritmos exatos com cplex (tem um para python)
4. Utilizar meta-heuristicas diferentes para selecionar as meta-features

### Duvidas

1. Utilizar mais de um meta-modelo durante a escolha das metafeatures?
2. 