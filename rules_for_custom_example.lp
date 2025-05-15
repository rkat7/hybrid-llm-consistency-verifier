% Rules for custom example test

% Define persons
person(john).
person(may).
person(jay).

% Rules for marriage
married(X) :- person(X), john(X).
married(X) :- person(X), may(X).
married(X) :- person(X), jay(X).

% Married to relationship
married_to(X,Y) :- person(X), person(Y), married(X), married(Y), X != Y.

% Identity rules for same_as relationship
same_as(X,X) :- person(X).  % Reflexivity
same_as(Y,X) :- same_as(X,Y).  % Symmetry
same_as(X,Z) :- same_as(X,Y), same_as(Y,Z).  % Transitivity

% Sameness propagates properties
john(X) :- john(Y), same_as(X,Y).
may(X) :- may(Y), same_as(X,Y).
jay(X) :- jay(Y), same_as(X,Y).

% Contradictory rules
:- john(X), jay(X), X != john, X != jay.  % John and Jay cannot be the same person
:- married_to(X,Y), same_as(X,Y).  % A person cannot be married to themselves

% Show directives
#show person/1.
#show married/1.
#show married_to/2.
#show john/1.
#show may/1.
#show jay/1.
#show same_as/2. 