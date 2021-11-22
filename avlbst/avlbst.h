#ifndef RBBST_H
#define RBBST_H

#include <iostream>
#include <exception>
#include <cstdlib>
#include <algorithm>
#include <memory>
#include "bst.h"

struct KeyError { };

/**
* A special kind of node for an AVL tree, which adds the balance as a data member, plus
* other additional helper functions. You do NOT need to implement any functionality or
* add additional data members or helper functions.
*/
template <typename Key, typename Value>
class AVLNode : public Node<Key, Value>
{
public:
    // Constructor/destructor.
    AVLNode(const Key& key, const Value& value, std::shared_ptr<AVLNode<Key, Value>> parent);
    virtual ~AVLNode();

    // Getter/setter for the node's height.
    char getBalance () const;
    void setBalance (char balance);
    void updateBalance(char diff);

    std::shared_ptr<AVLNode<Key, Value>> getParent_AVL() const;
    std::shared_ptr<AVLNode<Key, Value>> getLeft_AVL() const;
    std::shared_ptr<AVLNode<Key, Value>> getRight_AVL() const;

protected:
    // to store the balance of a given node
    char balance_;
};


    // -------------------------------------------------
    // Begin implementations for the AVLNode class.
    // -------------------------------------------------



// An explicit constructor to initialize the elements by calling the base class constructor and setting
// the color to red since every new node will be red when it is first inserted.

template<class Key, class Value>
AVLNode<Key, Value>::AVLNode(const Key& key, const Value& value, std::shared_ptr<AVLNode<Key, Value>> parent) :
    Node<Key, Value>(key, value, parent), balance_(0)
{

}

// A destructor which does nothing.
template<class Key, class Value>
AVLNode<Key, Value>::~AVLNode()
{

}

// A getter for the balance of a AVLNode.
template<class Key, class Value>
char AVLNode<Key, Value>::getBalance() const
{
    return balance_;
}

// A setter for the balance of a AVLNode.
template<class Key, class Value>
void AVLNode<Key, Value>::setBalance(char balance)
{
    balance_ = balance;
}

// Adds diff to the balance of a AVLNode.
template<class Key, class Value>
void AVLNode<Key, Value>::updateBalance(char diff)
{
    balance_ += diff;
}

// A separate getParent_AVL function other than the base class function due to covariant return types
template<class Key, class Value>
std::shared_ptr<AVLNode<Key, Value>> AVLNode<Key, Value>::getParent_AVL() const
{
    return std::static_pointer_cast< AVLNode<Key, Value>>(this->parent_);
}

// Similar getLeft_AVL function
template<class Key, class Value>
std::shared_ptr<AVLNode<Key, Value>> AVLNode<Key, Value>::getLeft_AVL() const
{
    return std::static_pointer_cast< AVLNode<Key, Value>>(this->left_);
}

// Similar getRight_AVL function
template<class Key, class Value>
std::shared_ptr<AVLNode<Key, Value>> AVLNode<Key, Value>::getRight_AVL() const
{
    return std::static_pointer_cast< AVLNode<Key, Value>>(this->right_);
}


// -----------------------------------------------
// End implementations for the AVLNode class.
// -----------------------------------------------


template <class Key, class Value>
class AVLTree : public BinarySearchTree<Key, Value>
{
public:
    
    void rotateLeft(std::shared_ptr<AVLNode<Key, Value>> p, std::shared_ptr<AVLNode<Key, Value>> n); //TODO
    void rotateRight(std::shared_ptr<AVLNode<Key, Value>> p, std::shared_ptr<AVLNode<Key, Value>> n); //TODO

    // Remember, AVL is a self-balancing BST
    // Resultant tree after the insert and remove function should be a balanced tree
    // Make appropriate calls to rotateLeft(...) and rotateRight(...)
    // in insert and remove for balancing the height of the AVLTree
    virtual void insert (const std::pair<const Key, Value> &new_item); // TODO
    virtual void remove(const Key& key);  // TODO
protected:
    // Helper function already provided to you.
    virtual void nodeSwap( std::shared_ptr<AVLNode<Key,Value>> n1, std::shared_ptr<AVLNode<Key,Value>> n2);
    
    void insertFix(std::shared_ptr<AVLNode<Key,Value>> p, std::shared_ptr<AVLNode<Key,Value>> n);
    void removeFix(std::shared_ptr<AVLNode<Key, Value>> p, int diff);
    // Add helper functions here
    // Consider adding functions like getBalance(...) given a key in the Tree
    // setBalance(...) given a key to a node and balance value, etc
    // You may implement a printRootAVL(...) 
    // using the printRoot() function from the BST implementation
};
// Pre condition: p is the parent of n 
// Post condition: p is the left child of n
template<class Key, class Value>
void AVLTree<Key, Value>::rotateLeft(std::shared_ptr<AVLNode<Key, Value>> p, std::shared_ptr<AVLNode<Key, Value>> n)
{
    // TODO
    if(p->getParent_AVL() != nullptr) {
        std::shared_ptr<AVLNode<Key, Value>> g = p->getParent_AVL();
        std::shared_ptr<AVLNode<Key, Value>> nleft = n->getLeft_AVL(); //n's left child will become p's right child
        if(g->getLeft_AVL() == p) {
            g->setLeft(n);
            n->setParent(g);
            p->setParent(n);
            n->setLeft(p);
            p->setRight(nleft);
            if(nleft != nullptr) {
                nleft->setParent(p);
            }
        } else {
            g->setRight(n);
            n->setParent(g);
            p->setParent(n);
            n->setLeft(p);
            p->setRight(nleft);
            if(nleft != nullptr) {
                nleft->setParent(p);
            }
        }
    } else { //If we are rotating at the root, we have to update the root
        std::shared_ptr<AVLNode<Key, Value>> nleft = n->getLeft_AVL();
        n->setLeft(p);
        p->setParent(n);
        p->setRight(nleft);
        if(nleft != nullptr) {
            nleft->setParent(p);
        }
        this->root_ = n;
        this->root_->setParent(nullptr);
    }
}

// Pre condition: p is the parent of n 
// Post condition: p is the right child of n
template<class Key, class Value>
void AVLTree<Key, Value>::rotateRight(std::shared_ptr<AVLNode<Key, Value>> p, std::shared_ptr<AVLNode<Key, Value>> n)
{
    // TODO
    if(p->getParent_AVL() != nullptr) {
        std::shared_ptr<AVLNode<Key, Value>> g = p->getParent_AVL();
        std::shared_ptr<AVLNode<Key, Value>> nright = n->getRight_AVL(); //n's right child will become p's left child
        if(g->getLeft_AVL() == p) {
            g->setLeft(n);
            n->setParent(g);
            p->setParent(n);
            n->setRight(p);
            p->setLeft(nright);
            if(nright != nullptr) {
                nright->setParent(p);
            }
        } else {
            g->setRight(n);
            n->setParent(g);
            p->setParent(n);
            n->setRight(p);
            p->setLeft(nright);
            if(nright != nullptr) {
                nright->setParent(p);
            }
        }
    } else { //If we are rotating at the root, we have to update the root
        std::shared_ptr<AVLNode<Key, Value>> nright = n->getRight_AVL();
        n->setRight(p);
        p->setParent(n);
        p->setLeft(nright);
        if(nright != nullptr) {
            nright->setParent(p);
        }
        this->root_ = n;
        this->root_->setParent(nullptr);
    }
}

template<class Key, class Value>
void AVLTree<Key, Value>::insert (const std::pair<const Key, Value> &new_item)
{
    // TODO
    std::shared_ptr<Node<Key, Value>> find = this->internalFind(new_item.first); //find using BST find
    if(find == nullptr) {
        if(this->root_ == nullptr) {
            this->root_ = std::make_shared<AVLNode<Key, Value>>(new_item.first, new_item.second, nullptr); //make root
            std::shared_ptr<AVLNode<Key, Value>> rootAVL = std::static_pointer_cast<AVLNode<Key, Value>>(this->root_);
            rootAVL->setBalance(0); //set its balance to 0
        } else {
            std::shared_ptr<AVLNode<Key, Value>> rootAVL = std::static_pointer_cast<AVLNode<Key, Value>>(this->root_);
            std::shared_ptr<AVLNode<Key, Value>> curr = rootAVL;
            std::shared_ptr<AVLNode<Key, Value>> temp;
            while(curr != nullptr) { //Similar to BST except using getRight_AVL() and getLeft_AVL()
                if(new_item.first > curr->getKey()) {
                    if(curr->getRight_AVL() == nullptr) {
                        temp = std::make_shared<AVLNode<Key, Value>>(new_item.first, new_item.second, curr);
                        curr->setRight(temp);
                        break;
                    }
                    curr = curr->getRight_AVL();
                } else {
                    if(curr->getLeft_AVL() == nullptr) {
                        temp = std::make_shared<AVLNode<Key, Value>>(new_item.first, new_item.second,
                            curr);
                        curr->setLeft(temp);
                        break;
                    }
                    curr = curr->getLeft_AVL();
                }
            }
            temp->setBalance(0);

            if(temp->getParent_AVL()->getBalance() == -1) { //update balances of parent
                temp->getParent_AVL()->setBalance(0);
            } else if(temp->getParent_AVL()->getBalance() == 1) {
                temp->getParent_AVL()->setBalance(0);
            } else if(temp->getParent_AVL()->getBalance() == 0) {
                if(BinarySearchTree<Key, Value>::isLeftChild(temp)) {
                    temp->getParent_AVL()->setBalance(-1);
                    insertFix(temp->getParent_AVL(), temp); //call insertFix on the newly inserted element's parent
                } else {
                    temp->getParent_AVL()->setBalance(1);
                    insertFix(temp->getParent_AVL(), temp);
                }
            }
        }
    } else {
        std::shared_ptr<AVLNode<Key, Value>> change = std::static_pointer_cast<AVLNode<Key, Value>>(find);
        change->setValue(new_item.second);
    }
}

template <class Key, class Value>
void AVLTree<Key, Value>::insertFix(std::shared_ptr<AVLNode<Key, Value>> p, std::shared_ptr<AVLNode<Key, Value>> n) {

    if(p == nullptr) {
        return;
    }
    
    std::shared_ptr<AVLNode<Key, Value>> g;
    
    if(p->getParent_AVL() == nullptr) {
        return;
    } else {
        g = p->getParent_AVL();
    }
    if(g->getLeft_AVL() == p) { //Two cases, left and right
        g->updateBalance(-1); //We have added a node on the left subtree of g
        if(g->getBalance() == 0) {
            return;
        } else if(g->getBalance() == -1) {
            insertFix(g, p);
        } else if(g->getBalance() == -2) {
            if(p->getLeft_AVL() == n) {
                rotateRight(g, p);
                g->setBalance(0);
                p->setBalance(0);
            } else { //Zig-zag case with three sub-cases
                rotateLeft(p, n);
                rotateRight(g, n);
                if(n->getBalance() == -1) {
                    p->setBalance(0);
                    g->setBalance(1);
                    n->setBalance(0);
                } else if(n->getBalance() == 0) {
                    p->setBalance(0);
                    g->setBalance(0);
                    n->setBalance(0);
                } else if(n->getBalance() == 1) {
                    p->setBalance(-1);
                    g->setBalance(0);
                    n->setBalance(0);
                }
            }
        }
    } else {
        g->updateBalance(1); //We have added a node on the right substree of g
        if(g->getBalance() == 0) {
            return;
        } else if(g->getBalance() == 1) {
            insertFix(g, p);
        } else if(g->getBalance() == 2) {
            if(p->getRight_AVL() == n) {
                rotateLeft(g, p);
                g->setBalance(0);
                p->setBalance(0);
            } else { //Zig-zag case with three sub cases
                rotateRight(p, n);
                rotateLeft(g, n);
                if(n->getBalance() == -1) {
                    p->setBalance(1);
                    g->setBalance(0);
                    n->setBalance(0);
                } else if(n->getBalance() == 0) {
                    p->setBalance(0);
                    g->setBalance(0);
                    n->setBalance(0);
                } else if(n->getBalance() == 1) {
                    p->setBalance(0);
                    g->setBalance(-1);
                    n->setBalance(0);
                }
            }
        }
    }
}

template<class Key, class Value>
void AVLTree<Key, Value>::remove(const Key& key) {
    // TODO
    bool left = false;
    this->printRoot(this->root_);
    std::shared_ptr<Node<Key, Value>> n = BinarySearchTree<Key, Value>::internalFind(key);
    std::shared_ptr<AVLNode<Key, Value>> find = std::static_pointer_cast<AVLNode<Key, Value>>(n);
    if(find != nullptr) {
        std::shared_ptr<AVLNode<Key, Value>> p = find->getParent_AVL();
        if(find->getLeft_AVL() == nullptr && find->getRight_AVL() == nullptr) { //If node to be removed has no children
            if(n != this->root_) {
                if(BinarySearchTree<Key, Value>::isLeftChild(find)) {
                    p->setLeft(nullptr);
                    left = true;
                } else {
                    p->setRight(nullptr);
                }
            } else {
                this->root_->setLeft(nullptr);
                this->root_->setRight(nullptr);
                this->root_->setParent(nullptr);
                this->root_ = nullptr;
                return;
            }
            find->setLeft(nullptr);
            find->setRight(nullptr);
            find->setParent(nullptr);
            find.reset();
        } else if(find->getLeft_AVL() != nullptr && find->getRight_AVL() != nullptr) { //If node has two children, swap with predescessor
            std::shared_ptr<Node<Key, Value>> pred = BinarySearchTree<Key, Value>::predecessor(find);
            std::shared_ptr<AVLNode<Key, Value>> predCast = std::static_pointer_cast<AVLNode<Key, Value>>(pred);
            AVLTree<Key, Value>::nodeSwap(predCast, find);
            std::shared_ptr<AVLNode<Key, Value>> parent = find->getParent_AVL();
            if(BinarySearchTree<Key, Value>::isLeftChild(find)) {
                left = true;
                parent->setLeft(nullptr);
            } else {
                parent->setRight(nullptr);
            }
            if(pred == this->root_) {
                this->root_ = pred;
            }
            find->setLeft(nullptr);
            find->setRight(nullptr);
            find->setParent(nullptr);
            find.reset();
            p = parent;
        } else if(find->getLeft_AVL() != nullptr) { //If node has one child, just promote the child
            std::shared_ptr<AVLNode<Key, Value>> leftChild = find->getLeft_AVL();
            std::shared_ptr<AVLNode<Key, Value>> parent = find->getParent_AVL();
            if(parent != nullptr) {
                if(BinarySearchTree<Key, Value>::isLeftChild(find)) {
                    left = true;
                    parent->setLeft(leftChild);
                } else {
                    parent->setRight(leftChild);
                }
                leftChild->setParent(parent);
            } else {
                leftChild->setParent(nullptr);
                this->root_ = leftChild;
            }
            find->setLeft(nullptr);
            find->setRight(nullptr);
            find->setParent(nullptr);
            find.reset();
            p = parent;
        } else if(find->getRight_AVL() != nullptr) {
            std::shared_ptr<AVLNode<Key, Value>> rightChild = find->getRight_AVL();
            std::shared_ptr<AVLNode<Key, Value>> parent = find->getParent_AVL();
            
            if(parent != nullptr) {
                if(BinarySearchTree<Key, Value>::isLeftChild(find)) {
                    left = true;
                    parent->setLeft(rightChild);
                } else {
                    parent->setRight(rightChild);
                }
                rightChild->setParent(parent);
            } else {
                rightChild->setParent(nullptr);
                this->root_ = rightChild;
            }
            find->setLeft(nullptr);
            find->setRight(nullptr);
            find->setParent(nullptr);
            find.reset();
            p = parent;
        }

        int diff = 0;
        if(p != nullptr) {
            if(left) {
                diff = 1;
            } else {
                diff = -1;
            }
            removeFix(p, diff); //call removeFix (with the difference to be added to the parent's balance)
        }
    }
}

template<class Key, class Value>
void AVLTree<Key, Value>::removeFix(std::shared_ptr<AVLNode<Key, Value>> n, int diff) {
    if(n == nullptr) {
        return;
    }
    std::shared_ptr<AVLNode<Key, Value>> p = n->getParent_AVL();
    int nextDiff = 0;
    if(p != nullptr) { //calculate next recursive call for when needed
        if(p->getRight_AVL() == n) {
            nextDiff = -1;
        } else {
            nextDiff = 1;
        }
    }

    BinarySearchTree<Key, Value>::printRoot(this->root_);
    if(diff == -1) { //two cases
        if(n->getBalance() + diff == -2) {
            std::shared_ptr<AVLNode<Key, Value>> c = n->getLeft_AVL();
            if(c->getBalance() == -1) {
                rotateRight(n, c);
                n->setBalance(0);
                c->setBalance(0);
            } else if(c->getBalance() == 0) {
                rotateRight(n, c);
                n->setBalance(-1);
                c->setBalance(1);
            } else if(c->getBalance() == 1) { //zig-zag case
                std::shared_ptr<AVLNode<Key, Value>> g = c->getRight_AVL();
                rotateLeft(c, g);
                rotateRight(n, g);
                if(g->getBalance() == 1) {
                    n->setBalance(0);
                    c->setBalance(-1);
                    g->setBalance(0);
                } else if(g->getBalance() == 0) {
                    n->setBalance(0);
                    c->setBalance(0);
                    g->setBalance(0);
                } else if(g->getBalance() == -1) {
                    n->setBalance(1);
                    c->setBalance(0);
                    g->setBalance(0);
                }
            }
        } else if(n->getBalance() + diff == -1) {
            n->setBalance(-1);
        } else if(n->getBalance() + diff == 0) {
            n->setBalance(0);
            if(p != nullptr) {
                removeFix(p, nextDiff); //call next parent
            }
        }
    } else if(diff == 1) {
        if(n->getBalance() + diff == 2) {
            std::shared_ptr<AVLNode<Key, Value>> c = n->getRight_AVL();
            if(c->getBalance() == 1) {
                rotateLeft(n, c);
                n->setBalance(0);
                c->setBalance(0);
            } else if(c->getBalance() == 0) {
                rotateLeft(n, c);
                n->setBalance(1);
                c->setBalance(-1);
            } else if(c->getBalance() == -1) { //zig-zag case
                std::shared_ptr<AVLNode<Key, Value>> g = c->getLeft_AVL();
                rotateRight(c, g);
                rotateLeft(n, g);
                if(g->getBalance() == -1) {
                    n->setBalance(0);
                    c->setBalance(1);
                    g->setBalance(0);
                } else if(g->getBalance() == 0) {
                    n->setBalance(0);
                    c->setBalance(0);
                    g->setBalance(0);
                } else if(g->getBalance() == 1) {
                    n->setBalance(-1);
                    c->setBalance(0);
                    g->setBalance(0);
                }
            } else if(n->getBalance() + diff == 1) {
                n->setBalance(1);
            } else if(n->getBalance() + diff == 0) {
                n->setBalance(0);
                if(p != nullptr) {
                    removeFix(p, nextDiff); //call next parent
                }
            }
        }
    }
    
}
// Function already completed for you
template<class Key, class Value>
void AVLTree<Key, Value>::nodeSwap( std::shared_ptr<AVLNode<Key,Value>> n1, std::shared_ptr<AVLNode<Key,Value>> n2)
{
    BinarySearchTree<Key, Value>::nodeSwap(n1, n2);
    char tempB = n1->getBalance();
    n1->setBalance(n2->getBalance());
    n2->setBalance(tempB);
}

#endif
