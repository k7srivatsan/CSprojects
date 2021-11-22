#ifndef BST_H
#define BST_H

#include <iostream>
#include <exception>
#include <cstdlib>
#include <utility>
#include <algorithm>
#include <memory>
#include <math.h>
#include <set>

// A templated class for a Node in a search tree.
// The getters for parent/left/right are virtual so
// that they can be overridden for future kinds of
// search trees, such as Red Black trees, Splay trees,
// and AVL trees.
 
// Think carefully when implementing ths BST class functionalities
// as you would be using them for the next part of the assignment.
template <typename Key, typename Value>
class Node
{
public:
    Node(const Key& key, const Value& value, std::shared_ptr<Node<Key, Value>> parent);
    virtual ~Node();

    const std::pair<const Key, Value>& getItem() const;
    std::pair<const Key, Value>& getItem();
    const Key& getKey() const;
    const Value& getValue() const;
    Value& getValue();

    virtual std::shared_ptr<Node<Key, Value>> getParent() const;
    virtual std::shared_ptr<Node<Key, Value>> getLeft() const;
    virtual std::shared_ptr<Node<Key, Value>> getRight() const;

    void setParent(std::shared_ptr<Node<Key, Value>> parent);
    void setLeft(std::shared_ptr<Node<Key, Value>> left);
    void setRight(std::shared_ptr<Node<Key, Value>> right);
    void setValue(const Value &value);

protected:
    std::pair<const Key, Value> item_;
    std::shared_ptr<Node<Key, Value>> parent_;
    std::shared_ptr<Node<Key, Value>> left_;
    std::shared_ptr<Node<Key, Value>> right_;
};

/*
  -----------------------------------------
  Begin implementations for the Node class.
  -----------------------------------------
*/


// Explicit constructor for a node.
template<typename Key, typename Value>
Node<Key, Value>::Node(const Key& key, const Value& value, std::shared_ptr<Node<Key, Value>> parent) :
    item_(key, value),
    parent_(parent),
    left_(NULL),
    right_(NULL)
{

}

/*
* Destructor, which does not need to do anything since the pointers inside of a node
* are only used as references to existing nodes. The nodes pointed to by parent/left/right
* are freed within the deleteAll() helper method in the BinarySearchTree.
*/
template<typename Key, typename Value>
Node<Key, Value>::~Node()
{

}

/**
* A const getter for the item.
*/
template<typename Key, typename Value>
const std::pair<const Key, Value>& Node<Key, Value>::getItem() const
{
    return item_;
}

/**
* A non-const getter for the item.
*/
template<typename Key, typename Value>
std::pair<const Key, Value>& Node<Key, Value>::getItem()
{
    return item_;
}

/**
* A const getter for the key.
*/
template<typename Key, typename Value>
const Key& Node<Key, Value>::getKey() const
{
    return item_.first;
}

/**
* A const getter for the value.
*/
template<typename Key, typename Value>
const Value& Node<Key, Value>::getValue() const
{
    return item_.second;
}

/**
* A non-const getter for the value.
*/
template<typename Key, typename Value>
Value& Node<Key, Value>::getValue()
{
    return item_.second;
}

/**
* An implementation of the virtual function for retreiving the parent.
*/
template<typename Key, typename Value>
std::shared_ptr<Node<Key, Value>> Node<Key, Value>::getParent() const
{
    return parent_;
}

/**
* An implementation of the virtual function for retreiving the left child.
*/
template<typename Key, typename Value>
std::shared_ptr<Node<Key, Value>> Node<Key, Value>::getLeft() const
{
    return left_;
}

/**
* An implementation of the virtual function for retreiving the right child.
*/
template<typename Key, typename Value>
std::shared_ptr<Node<Key, Value>> Node<Key, Value>::getRight() const
{
    return right_;
}

/**
* A setter for setting the parent of a node.
*/
template<typename Key, typename Value>
void Node<Key, Value>::setParent(std::shared_ptr<Node<Key, Value>> parent)
{
    parent_ = parent;
}

/**
* A setter for setting the left child of a node.
*/
template<typename Key, typename Value>
void Node<Key, Value>::setLeft(std::shared_ptr<Node<Key, Value>> left)
{
    left_ = left;
}

/**
* A setter for setting the right child of a node.
*/
template<typename Key, typename Value>
void Node<Key, Value>::setRight(std::shared_ptr<Node<Key, Value>> right)
{
    right_ = right;
}

/**
* A setter for the value of a node.
*/
template<typename Key, typename Value>
void Node<Key, Value>::setValue(const Value& value)
{
    item_.second = value;
}

/*
  ---------------------------------------
  End implementations for the Node class.
  ---------------------------------------
*/

/**
* A templated unbalanced binary search tree.
*/
template <typename Key, typename Value>
class BinarySearchTree
{
public:
    BinarySearchTree(); //TODO
    virtual ~BinarySearchTree(); //TODO
    virtual void insert(const std::pair<const Key, Value>& keyValuePair); //TODO
    virtual void remove(const Key& key); //TODO
    void clear(); //TODO
    bool isBalanced() const; //TODO
    void print() const;
    bool empty() const;
public:
    
    // An internal iterator class for traversing the contents of the BST.
    // Complete the required implementations 
    // Additional functions are not necessary for this class.
    // Although, you may add them if you need any.
    class iterator  // TODO
    {
    public:
        iterator();

        std::pair<const Key,Value> & operator*() const;
        std::shared_ptr<std::pair<const Key,Value>> operator->() const;

        bool operator==(const iterator& rhs) const;
        bool operator!=(const iterator& rhs) const;

        iterator& operator++();

    protected:
        friend class BinarySearchTree<Key, Value>;
        iterator(std::shared_ptr<Node<Key,Value>> ptr);
        std::shared_ptr<Node<Key, Value>> current_;
    };

public:

    //Functions already completed for you
    iterator begin() const;
    iterator end() const;
    iterator find(const Key& key) const;

protected:
    // Mandatory helper functions you need to complete
    std::shared_ptr<Node<Key, Value>> internalFind(const Key& k) const; // TODO
    std::shared_ptr<Node<Key, Value>> getSmallestNode() const;  // TODO
    static std::shared_ptr<Node<Key, Value>> predecessor(std::shared_ptr<Node<Key, Value>> current); // TODO
    static std::shared_ptr<Node<Key, Value>> successor(std::shared_ptr<Node<Key, Value>> current); // TODO
    // Note:  static means these functions don't have a "this" pointer
    //        and instead just use the input argument.

    // Helper functions completed for you
    virtual void printRoot (std::shared_ptr<Node<Key, Value>> r) const;
    virtual void nodeSwap( std::shared_ptr<Node<Key,Value>> n1, std::shared_ptr<Node<Key,Value>> n2) ;


    // Add helper functions here
    // Consider adding simple helper functions like hasParent(...), isLeftChild(...), isRightChild(...)
    // Or functions like getHeight(...), removeInternal(...), etc 
    bool hasParent(std::shared_ptr<Node<Key, Value>> ptr) const;
    bool isLeftChild(std::shared_ptr<Node<Key, Value>> ptr) const;
    bool isRightChild(std::shared_ptr<Node<Key, Value>> ptr) const;
    int getHeight(std::shared_ptr<Node<Key, Value>> ptr) const;
    bool isBalancedHelper(std::shared_ptr<Node<Key, Value>> curr) const;
protected:
    std::shared_ptr<Node<Key, Value>> root_;
    // You should not add other data members
};

/*
--------------------------------------------------------------
Begin implementations for the BinarySearchTree::iterator class.
---------------------------------------------------------------
*/

// Explicit constructor that initializes an iterator with a given node pointer.
template<class Key, class Value>
BinarySearchTree<Key, Value>::iterator::iterator(std::shared_ptr<Node<Key,Value>> ptr) :
    current_(ptr)
{
    // TODO
    
}

// A default constructor that initializes the iterator to NULL.
template<class Key, class Value>
BinarySearchTree<Key, Value>::iterator::iterator() :
    current_(NULL)
{
    // TODO
}


//Provides access to the item.
template<class Key, class Value>
std::pair<const Key,Value> & BinarySearchTree<Key, Value>::iterator::operator*() const
{
    // TODO
    return current_->getItem();
}

// Provides access to the address of the item.
template<class Key, class Value>
std::shared_ptr<std::pair<const Key,Value>> BinarySearchTree<Key, Value>::iterator::operator->() const
{
    // TODO
    std::shared_ptr<std::pair<const Key, Value>> ptr = std::make_shared<std::pair<const Key, Value>>(current_->getItem());
    return ptr;
}


// Checks if 'this' iterator's internals have the same value as 'rhs'
template<class Key, class Value>
bool BinarySearchTree<Key, Value>::iterator::operator==( const BinarySearchTree<Key, Value>::iterator& rhs) const
{
    // TODO
    return (this->current_ == rhs.current_);
}

// Checks if 'this' iterator's internals have a different value as 'rhs'
template<class Key, class Value>
bool BinarySearchTree<Key, Value>::iterator::operator!=(const BinarySearchTree<Key, Value>::iterator& rhs) const
{
    // TODO
    return (this->current_ != rhs.current_);
}

// Advances the iterator's location using an in-order sequencing
template<class Key, class Value>
typename BinarySearchTree<Key, Value>::iterator& BinarySearchTree<Key, Value>::iterator::operator++()
{
    // TODO
    if(current_->getRight() != nullptr) { //We want successor if current has right child
        current_ = current_->getRight();
        while(current_->getLeft() != nullptr) {
            current_ = current_->getLeft();
        }
    } else { //Otherwise we traverse up
        if(current_->getParent() == nullptr) {
            current_ = nullptr;
        } else {
            std::shared_ptr<Node<Key, Value>> p = current_->getParent();
            while (p != nullptr && current_ == p->getRight()) {
                current_ = p;
                p = p->getParent();
            }
            current_ = p;
        }
    }
    return *this;
}

// -------------------------------------------------------------
// End implementations for the BinarySearchTree::iterator class.
// -------------------------------------------------------------


// -----------------------------------------------------
// Begin implementations for the BinarySearchTree class.
// -----------------------------------------------------

// Default constructor for a BinarySearchTree, which sets the root to NULL.
template<class Key, class Value>
BinarySearchTree<Key, Value>::BinarySearchTree()
{
    // TODO
    this->root_ = nullptr;
}

template<typename Key, typename Value>
BinarySearchTree<Key, Value>::~BinarySearchTree()
{
    // TODO
    this->clear();
}


// Returns true if tree is empty 
template<class Key, class Value>
bool BinarySearchTree<Key, Value>::empty() const
{
    return (root_ == nullptr);
}

// print the tree using the provided printRoot function
template<typename Key, typename Value>
void BinarySearchTree<Key, Value>::print() const
{
    printRoot(root_);
    std::cout << "\n";
}


// Returns an iterator to the "smallest" item in the tree
template<class Key, class Value>
typename BinarySearchTree<Key, Value>::iterator BinarySearchTree<Key, Value>::begin() const
{
    BinarySearchTree<Key, Value>::iterator begin(getSmallestNode());
    return begin;
}

// Returns an iterator whose value means INVALID
template<class Key, class Value>
typename BinarySearchTree<Key, Value>::iterator BinarySearchTree<Key, Value>::end() const
{
    BinarySearchTree<Key, Value>::iterator end(NULL);
    return end;
}


// Returns an iterator to the item with the given key, k 
// or the end iterator if k does not exist in the tree
template<class Key, class Value>
typename BinarySearchTree<Key, Value>::iterator BinarySearchTree<Key, Value>::find(const Key & k) const
{
    std::shared_ptr<Node<Key, Value>> curr = internalFind(k);
    BinarySearchTree<Key, Value>::iterator it(curr);
    return it;
}
template<class Key, class Value>
bool BinarySearchTree<Key, Value>::hasParent(std::shared_ptr<Node<Key, Value>> ptr) const {
    if(ptr == root_) {
        return false;
    }
    return true;
}

template<class Key, class Value>
bool BinarySearchTree<Key, Value>::isLeftChild(std::shared_ptr<Node<Key, Value>> ptr) const {
    if(!hasParent(ptr)) {
        return false;
    }
    return (ptr->getParent()->getLeft() == ptr);
}

template<class Key, class Value>
bool BinarySearchTree<Key, Value>::isRightChild(std::shared_ptr<Node<Key, Value>> ptr) const {
    if(!hasParent(ptr)) {
        return false;
    }
    return (ptr->getParent()->getRight() == ptr);
}

template<class Key, class Value>
int BinarySearchTree<Key, Value>::getHeight(std::shared_ptr<Node<Key, Value>> ptr) const {
    if(ptr == nullptr) {
        return 0;
    }
    int left = getHeight(ptr->getLeft());
    int right = getHeight(ptr->getRight());
    if(right > left) {
        return (right + 1);
    } else {
        return (left + 1);
    }
}

// An insert method to insert into a Binary Search Tree.
// If the key is already present in the tree,
// update the current value with the new value.
// The tree may not remain balanced when inserting.
template<class Key, class Value>
void BinarySearchTree<Key, Value>::insert(const std::pair<const Key, Value> &keyValuePair)
{
    // TODO
    std::shared_ptr<Node<Key, Value>> find = this->internalFind(keyValuePair.first);
    if(find == nullptr) {
        std::shared_ptr<Node<Key, Value>> curr = root_;
        while(curr != nullptr) {
            if(keyValuePair.first <= curr->getKey()) { //While it is less than current, we get left
                if(curr->getLeft() == nullptr) {
                    std::shared_ptr<Node<Key, Value>> temp = std::make_shared<Node<Key, Value>>(keyValuePair.first, keyValuePair.second, curr);
                    curr->setLeft(temp);
                    break;
                }
                curr = curr->getLeft();
            } else {
                if(curr->getRight() == nullptr) { //While it is more, we get right
                    std::shared_ptr<Node<Key, Value>> temp = std::make_shared<Node<Key, Value>>(keyValuePair.first, keyValuePair.second, curr);
                    curr->setRight(temp);
                    break;
                }
                curr = curr->getRight();
            }
            
        }
        if(curr == nullptr) {
            root_ = std::make_shared<Node<Key, Value>>(keyValuePair.first, keyValuePair.second, nullptr);
        }
    } else {
        find->setValue(keyValuePair.second); //If we find a matching key, we simply change the value
    }
}



// A remove method to remove a specific key from a Binary Search Tree.
// Does nothing if key not found.
// The tree may not remain balanced after removal.
template<typename Key, typename Value>
void BinarySearchTree<Key, Value>::remove(const Key& key)
{
    // TODO
    std::shared_ptr<Node<Key, Value>> find = this->internalFind(key);
    if(find != nullptr) {
        if(find->getLeft() == nullptr && find->getRight() == nullptr) { //Four cases: first is when node to be removed has no children
            if(hasParent(find)) {
                std::shared_ptr<Node<Key, Value>> parent = find->getParent();
                if(isLeftChild(find)) {
                    parent->setLeft(nullptr);
                } else {
                    parent->setRight(nullptr);
                }
            } else {
                root_->setLeft(nullptr);
                root_->setRight(nullptr);
                root_->setParent(nullptr);
                root_ = nullptr;
                return;
            }
            find->setLeft(nullptr);
            find->setRight(nullptr);
            find->setParent(nullptr);
            find.reset();
        } else if(find->getLeft() != nullptr && find->getRight() != nullptr) { //If node has two children
            std::shared_ptr<Node<Key, Value>> pred = predecessor(find);
            nodeSwap(pred, find); //have to Swap
            std::shared_ptr<Node<Key, Value>> parent = find->getParent();
            if(isLeftChild(find)) {
                if(find->getLeft() != nullptr) {
                    parent->setLeft(find->getLeft());
                    find->getLeft()->setParent(parent);
                } else {
                    parent->setLeft(nullptr);
                }
            } else {
                if(find->getLeft() != nullptr) {
                    parent->setRight(find->getLeft());
                    find->getLeft()->setParent(parent);
                } else {
                    parent->setRight(nullptr);
                }
            }
            if(pred == root_) {
                root_ = pred;
            }
            find->setLeft(nullptr);
            find->setRight(nullptr);
            find->setParent(nullptr);
            find.reset();
        } else if(find->getLeft() != nullptr) { //If node has only left child
            std::shared_ptr<Node<Key, Value>> left = find->getLeft();
            std::shared_ptr<Node<Key, Value>> parent = find->getParent();
            if(parent != nullptr) {
                if(isLeftChild(find)) {
                    parent->setLeft(left);
                } else {
                    parent->setRight(left);
                }
                left->setParent(parent);
            } else {
                left->setParent(nullptr);
                root_ = left;
            }
            find->setLeft(nullptr);
            find->setRight(nullptr);
            find->setParent(nullptr);
            find.reset();
        } else if(find->getRight() != nullptr) { //If node has only right child
            std::shared_ptr<Node<Key, Value>> right = find->getRight();
            std::shared_ptr<Node<Key, Value>> parent = find->getParent();
            
            if(parent != nullptr) {
                if(isLeftChild(find)) {
                    parent->setLeft(right);
                } else {
                    parent->setRight(right);
                }
                right->setParent(parent);
            } else {
                right->setParent(nullptr);
                root_ = right;
            }
            find->setLeft(nullptr);
            find->setRight(nullptr);
            find->setParent(nullptr);
            find.reset();
        }
    }
}

template<class Key, class Value>
std::shared_ptr<Node<Key, Value>> BinarySearchTree<Key, Value>::successor(std::shared_ptr<Node<Key, Value>> current)
{
    // TODO
    std::shared_ptr<Node<Key, Value>> curr = current->getRight();
    if(curr == nullptr) {
        return nullptr;
    }
    while(curr->getLeft() != nullptr) {
        curr = curr->getLeft();
    }
    return curr;
}

template<class Key, class Value>
std::shared_ptr<Node<Key, Value>> BinarySearchTree<Key, Value>::predecessor(std::shared_ptr<Node<Key, Value>> current)
{
    // TODO
    std::shared_ptr<Node<Key, Value>> curr = current->getLeft();
    if(curr == nullptr) {
        return nullptr;
    }
    while(curr->getRight() != nullptr) {
        curr = curr->getRight();
    }
    return curr;
}

// A method to remove all contents of the tree and
// reset the values in the tree for use again.
template<typename Key, typename Value>
void BinarySearchTree<Key, Value>::clear()
{
    // TODO
    std::shared_ptr<Node<Key, Value>> curr = root_;
    if(curr == nullptr) {
        return;
    }
    while(true) { //Iterative method to clear the tree using post-order traversal
        if(curr->getLeft() != nullptr) {
            curr = curr->getLeft();
        } else if(curr->getRight() != nullptr) {
            curr = curr->getRight();
        } else {
            if(hasParent(curr)) {
                std::shared_ptr<Node<Key, Value>> parent = curr->getParent();
                curr->setLeft(nullptr);
                curr->setRight(nullptr);
                if(isLeftChild(curr)) { //Based on if node to be removed is left child or right child, have to modify parent's child ptr
                    curr->getParent()->setLeft(nullptr);
                } else {
                    curr->getParent()->setRight(nullptr);
                }
                curr->setParent(nullptr);
                curr.reset();
                curr = parent;
            } else {
                root_->setLeft(nullptr);
                root_->setRight(nullptr);
                root_ = nullptr;
                break;
            }
        }
    }
}


// A helper function to find the smallest node in the tree.
template<typename Key, typename Value>
std::shared_ptr<Node<Key, Value>> BinarySearchTree<Key, Value>::getSmallestNode() const
{
    // TODO
    std::shared_ptr<Node<Key, Value>> curr = root_;
    if(root_ == nullptr) {
        return nullptr;
    }
    while(curr->getLeft() != nullptr) {
        curr = curr->getLeft();
    }
    return curr;
}

// Helper function to find a node with given key, k and
// return a pointer to it or nullptr if no item with that key exists
template<typename Key, typename Value>
std::shared_ptr<Node<Key, Value>> BinarySearchTree<Key, Value>::internalFind(const Key& key) const
{
    // TODO
    std::shared_ptr<Node<Key, Value>> curr = root_;
    while(true) {
        if(curr != nullptr) {
            if(key > curr->getKey()) {
                curr = curr->getRight();
            } else if(key < curr->getKey()) {
                curr = curr->getLeft();
            } else {
                if(curr->getKey() == key) {
                    return curr;
                }
            }
        } else {
            return nullptr; //If we came to a nullptr, we couldn't find it
        }
    }
}

// Return true iff the BST is balanced. 
// You may use additional helper functions
template<typename Key, typename Value>
bool BinarySearchTree<Key, Value>::isBalanced() const
{
    // TODO
    return isBalancedHelper(root_);
}

template<typename Key, typename Value>
bool BinarySearchTree<Key, Value>::isBalancedHelper(std::shared_ptr<Node<Key, Value>> curr) const { //Recursive helper to see if tree is balanced
    int left;
    int right;

    if(curr == nullptr) {
        return true;
    }
    left = getHeight(curr->getLeft());
    right = getHeight(curr->getRight());

    if((left - right <= 1 && left - right >= -1) && isBalancedHelper(curr->getLeft()) && isBalancedHelper(curr->getRight())) {
        return true;
        
    }
    return false;
}
// Function already implemented for you
template<typename Key, typename Value>
void BinarySearchTree<Key, Value>::nodeSwap( std::shared_ptr<Node<Key,Value>> n1, std::shared_ptr<Node<Key,Value>> n2)
{
    if((n1 == n2) || (n1 == NULL) || (n2 == NULL) ) {
        return;
    }
    std::shared_ptr<Node<Key, Value>> n1p = n1->getParent();
    std::shared_ptr<Node<Key, Value>> n1r = n1->getRight();
    std::shared_ptr<Node<Key, Value>> n1lt = n1->getLeft();
    bool n1isLeft = false;
    if(n1p != NULL && (n1 == n1p->getLeft())) n1isLeft = true;
    std::shared_ptr<Node<Key, Value>> n2p = n2->getParent();
    std::shared_ptr<Node<Key, Value>> n2r = n2->getRight();
    std::shared_ptr<Node<Key, Value>> n2lt = n2->getLeft();
    bool n2isLeft = false;
    if(n2p != NULL && (n2 == n2p->getLeft())) n2isLeft = true;


    std::shared_ptr<Node<Key, Value>> temp;
    temp = n1->getParent();
    n1->setParent(n2->getParent());
    n2->setParent(temp);

    temp = n1->getLeft();
    n1->setLeft(n2->getLeft());
    n2->setLeft(temp);

    temp = n1->getRight();
    n1->setRight(n2->getRight());
    n2->setRight(temp);

    if( (n1r != NULL && n1r == n2) ) {
        n2->setRight(n1);
        n1->setParent(n2);
    }
    else if( n2r != NULL && n2r == n1) {
        n1->setRight(n2);
        n2->setParent(n1);

    }
    else if( n1lt != NULL && n1lt == n2) {
        n2->setLeft(n1);
        n1->setParent(n2);

    }
    else if( n2lt != NULL && n2lt == n1) {
        n1->setLeft(n2);
        n2->setParent(n1);

    }


    if(n1p != NULL && n1p != n2) {
        if(n1isLeft) n1p->setLeft(n2);
        else n1p->setRight(n2);
    }
    if(n1r != NULL && n1r != n2) {
        n1r->setParent(n2);
    }
    if(n1lt != NULL && n1lt != n2) {
        n1lt->setParent(n2);
    }

    if(n2p != NULL && n2p != n1) {
        if(n2isLeft) n2p->setLeft(n1);
        else n2p->setRight(n1);
    }
    if(n2r != NULL && n2r != n1) {
        n2r->setParent(n1);
    }
    if(n2lt != NULL && n2lt != n1) {
        n2lt->setParent(n1);
    }


    if(this->root_ == n1) {
        this->root_ = n2;
    }
    else if(this->root_ == n2) {
        this->root_ = n1;
    }

}

/**
 * Lastly, we are providing you with a print function,
   BinarySearchTree::printRoot().
   Just call it with a node to start printing at, e.g:
   this->printRoot(this->root_) // or any other node pointer

   It will print up to 5 levels of the tree rooted at the passed node,
   in ASCII graphics format.
   We hope it will make debugging easier!
  */

// include print function (in its own file because it's fairly long)
#include "print_bst.h"


// ---------------------------------------------------
// End implementations for the BinarySearchTree class.
// ---------------------------------------------------


#endif
