% domain_rules.lp

% every bird flies unless flagged as exception
flies(X) :- bird(X), not exception(X).

% you cannot be both married AND single
:- married(X), single(X).

% you cannot both fly AND not_fly
:- flies(X), not_flies(X).

% only show these predicates in the answer set
#show married/1.
#show single/1.
#show flies/1.
#show not_flies/1.
