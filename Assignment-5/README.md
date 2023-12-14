[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/whVWeVGo)
# Assignment 3: Vector Space Query
**Westmont College Fall 2023**

**CS 128 Information Retrieval and Big Data**

*Assistant Professor* Mike Ryu (mryu@westmont.edu) 

## Autor Information
* **Name(s)**: Silas Ten Elshof
* **Email(s)**: stenelshof@westmont.edu

## Problem Description

Create a type of ranked retrieval that can be used to rank the results of president's Inaugural Speeches speeches based off of which ones are most relevant to the query terms. The scores will be determined using a vector space model and cosine similarities. The problem aslo involves writing good tests to cover all of the logic of the code that I wrote as well as covering obsure inputs. 

## Description of the Solution

First I implemented methods to perform on vectors. This involved calculating euclidean norms, dot products, and cosine similarities for use in ranking later. Then I implemented methods for documents. This included a method to get rid of any specified unwanted words, stem all of the words using SnowBall stemmer, and calculate the term frequency for a given term in the document. Then I implemented methods for corpuses. This included a method to return all of the unique words in the corpus and assigning an index to each. Also a method to check whether a term is present in a given document. Also a method to compute tf-idf scores using log(tf)*log(corpus size/df). Also a method to put all of the scores for a given document into a vector. Finally a method to create a matrix with the doc title and its vector with the scores. I also wrote test cases to cover obscure inputs

## Key Takeaways

I learned a lot about writing tests. I did not have much experience with this aspect of coding before this. I learned that writing test cases first can be helpful for the actually coding it out later because writing the tests forces you to understand the logic and what is supposed to happen in the code. It was also helpful to use my tests and write new ones as I was trying to get the logic right for my code.
