%— Repair (abductive) program —%

% Choice: keep or drop each extracted fact
{ keep(F); drop(F) } :- fact(F).

% Re-define facts only if kept
F :- fact(F), keep(F).

% Import the original extracted facts
#include "facts.lp".

% Re-use your domain constraints:
:- married(X), single(X).
:- flies(X), not_flies(X).

% Show which facts we choose to drop
#show drop/1.
