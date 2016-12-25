% Filename: Node.m
% Purpose: Declaration of node
classdef Node < handle
   properties
       % store the decision rule at each node
       pt_dm1
       pt_dm2
       % indicate whether this node is a 'leaf'
       isLeaf
       % store the posterior probability of each class
       lambda
       % connection to the child nodes
       childNode_left
       childNode_center
       childNode_right
   end
   methods
       % constructors
       function obj = Node()
             obj.isLeaf = 0;
       end 
   end
end