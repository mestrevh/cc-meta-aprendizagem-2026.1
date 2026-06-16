# Seleção de Meta-features e seu Efeito na Performance de Sistemas de Meta-aprendizagem

**Autores:** João Pedro Nunes Magalhães, Victor Hugo Silva Ângelo  
**Instituição:** Instituto de Computação (IC) -- Universidade Federal de Alagoas (UFAL)

Este diretório armazena a pesquisa acadêmica e a documentação associada ao projeto focado na otimização de sistemas de meta-aprendizagem por meio da seleção rigorosa de meta-features. O projeto explora o impacto da alta dimensionalidade no treinamento de metamodelos e propõe soluções ancoradas na intersecção entre Aprendizado de Máquina (*Machine Learning*) e Pesquisa Operacional.

## Resumo do Artigo

O artigo aborda os desafios práticos inerentes ao *Automated Machine Learning* (AutoML), com ênfase na mitigação da maldição da dimensionalidade e dos prejuízos causados pela redundância informacional e multicolinearidade provenientes da extração massiva de características de conjuntos de dados. Balizado pelo teorema *No Free Lunch*, o estudo foca em otimizar a recomendação de algoritmos preditivos. 

Para a seleção de meta-features, o trabalho avalia três arquiteturas metodológicas principais:

- **Métodos Exatos:** Otimização matemática via Programação Linear Inteira (PLI) através do *solver* CBC. Compreende o Problema de Seleção do Melhor Subconjunto, o Problema da Mochila com Penalidade e o Problema da Mochila Quadrática Linearizada, visando a extração exata do subconjunto ótimo.
- **Métodos Aproximados:** Modelagem meta-heurística ancorada em Algoritmos Genéticos (AG). Engloba a inicialização orientada pela Teoria da Informação (*Mutual Information*), mecanismos de busca local e restrições de cardinalidade (AG Esparso).
- **Abordagem Híbrida (*Seeded GA*):** Convergência estrutural que injeta os arranjos validados pelos modelos matemáticos exatos como indivíduos de elite na população inicial evolutiva, congregando a garantia formal de otimalidade à ampla capacidade topográfica de exploração inerente ao algoritmo estocástico.

Os resultados demonstram analítica e empiricamente a estabilização da curva preditiva ocasionada pela filtragem das características no nível meta. Destaca-se a Abordagem Híbrida, capaz de atingir expressivos 71,05% de assertividade na recomendação de algoritmos utilizando um limite restrito de 18 meta-features (o que corresponde a menos de 2% do repositório inicial extraído), ponderando adequadamente a eficácia preditiva contra o respectivo ônus computacional processual.

## Estrutura do Diretório

O diretório encontra-se estruturado para facilitar o acesso aos recursos do artigo e do código operacional:

- `docs/`: Documentação externa, literatura adicional e materiais de suporte.
- `lattex/`: Arquivos-fonte do artigo acadêmico redigido em LaTeX. Inclui o documento principal (`lattex/main/article.tex`), o respectivo pacote bibliográfico no formato BibTeX (`lattex/main/sbc-template.bib`), além dos arquivos de figuras e gráficos incorporados aos resultados da pesquisa.
- `src/`: Códigos-fonte e *scripts* de modelagem computacional desenvolvidos na linguagem Python. Abriga os módulos para a execução das abordagens exatas, aproximadas e híbridas, além de todo o *pipeline* de pré-processamento e validação cruzada do sistema de meta-aprendizagem.