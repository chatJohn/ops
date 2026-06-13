/*
简单实现一个hash map<int, string>
*/
#include<iostream>
#include<string>
using namespace std;

struct Node{
    int key;
    string value;
    Node* next;
    Node(int k, string v): key(k), value(v), next(nullptr){}
};
template<int Capacity>
class HashMap{
public:
    HashMap(){
        for(int i = 0; i < Capacity; i++){
            buckets[i] = nullptr;
        }
    }
    void put(int key, string value){
        int idx = hash(key);
        Node* cur = buckets[idx];
        while(cur){
            if(cur->key == key){
                cur->value = value;
                return;
            }
            cur = cur->next;
        }
        Node* new_node = new Node(key, value);
        new_node->next = buckets[idx];
        buckets[idx] = new_node;
    }
    string get(int key){
        int idx = hash(key);
        Node* cur = buckets[idx];
        while(cur){
            if(cur->key == key){
                return cur->value;
            }
            cur = cur->next;
        }
        return "";
    }
    bool constains(int key){
        int idx = hash(key);
        Node* cur = buckets[idx];
        while(cur){
            if(cur->key == key){
                return true;
            }
            cur = cur->next;
        }
        return false;
    }
    void remove(int key){
        int idx = hash(key);
        Node* cur = buckets[idx];
        Node* pre = nullptr;
        while(cur){
            if(cur->key == key){
                if(pre){
                    pre->next = cur->next;
                }else{
                    buckets[idx] = cur->next;
                }
                return;
            }
            pre = cur;
            cur = cur->next;
        }
    }
private:
    Node* buckets[Capacity];
    int hash(int key){
        return (abs(key) % Capacity);
    }
};