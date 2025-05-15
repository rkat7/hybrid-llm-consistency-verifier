% Strong inconsistency rules

% Define persons
person(john).
person(may).
person(jay).

% Rules for marriage
married(X) :- person(X), john(X).
married(X) :- person(X), may(X).
married(X) :- person(X), jay(X).

% Identity rules for same_as
john(may) :- same_as(john,may).
may(john) :- same_as(john,may).

% Direct inconsistency rule: john and may cannot be the same person
:- same_as(john,may).

% Show directives
#show person/1.
#show married/1.
#show john/1.
#show may/1.
#show jay/1.
#show same_as/2. 