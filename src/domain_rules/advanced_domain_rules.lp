% Advanced Domain Rules with Complex Relationships

% === Basic Facts ===
person(john).
person(may).
person(jay).
person(bob).
person(alice).
person(emma).

% === Basic Predicates ===
% Genders
male(john).
male(jay).
male(bob).
female(may).
female(alice).
female(emma).

% === Relationships ===
% Marriage rules
can_marry(X,Y) :- person(X), person(Y), X != Y, not related(X,Y).
married(X) :- married_to(X,Y).
married(Y) :- married_to(X,Y).

% Marriage constraints
:- married_to(X,Y), not can_marry(X,Y).
:- married_to(X,Y), married_to(X,Z), Y != Z.  % No polygamy
:- married_to(X,Y), same_as(X,Y).  % Can't marry yourself

% Family relationships
parent_of(john, bob).  % John is Bob's parent
parent_of(may, bob).   % May is Bob's parent
parent_of(jay, emma).  % Jay is Emma's parent
parent_of(alice, emma). % Alice is Emma's parent

% Derived family relationships
child_of(X,Y) :- parent_of(Y,X).
related(X,Y) :- parent_of(X,Y).
related(X,Y) :- child_of(X,Y).
related(X,Y) :- parent_of(Z,X), parent_of(Z,Y), X != Y.  % Siblings
related(X,Y) :- related(Y,X).  % Symmetry

% === Identity Rules ===
% Identity is reflexive, symmetric, and transitive
same_as(X,X) :- person(X).  
same_as(Y,X) :- same_as(X,Y).
same_as(X,Z) :- same_as(X,Y), same_as(Y,Z).

% Identity transfers properties
male(Y) :- male(X), same_as(X,Y).
female(Y) :- female(X), same_as(X,Y).
parent_of(Y,Z) :- parent_of(X,Z), same_as(X,Y).
child_of(Y,Z) :- child_of(X,Z), same_as(X,Y).

% === Contradictions and Constraints ===
% Can't be both male and female
:- male(X), female(X).

% Can't be your own parent or child
:- parent_of(X,X).
:- child_of(X,X).

% Different identity constraints for different people
:- same_as(john,jay).  % John and Jay can't be the same person
:- same_as(bob,emma).  % Bob and Emma can't be the same person

% === Occupation and Location Rules ===
% Occupations
occupation(john, doctor).
occupation(may, engineer).
occupation(jay, teacher).
occupation(alice, doctor).

% Location constraints
lives_in(john, newyork).
lives_in(may, newyork).
lives_in(jay, boston).
lives_in(alice, boston).
lives_in(emma, boston).
lives_in(bob, chicago).

% Couples usually live in the same place
:- married_to(X,Y), lives_in(X,Z), lives_in(Y,W), Z != W.

% Children live with parents until they're adults
minor(bob).
minor(emma).
:- minor(X), parent_of(Y,X), lives_in(Y,Z), not lives_in(X,Z).

% === Show directives ===
#show person/1.
#show male/1.
#show female/1.
#show married/1.
#show married_to/2.
#show parent_of/2.
#show child_of/2.
#show related/2.
#show same_as/2.
#show occupation/2.
#show lives_in/2.
#show minor/1. 