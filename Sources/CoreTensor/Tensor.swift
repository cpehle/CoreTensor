//
//  Tensor.swift
//  CoreTensor
//
//  Copyright 2016-2017 The DLVM Team.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

/// Tensor
public struct Tensor<UnitType> : TensorProtocol {
    public typealias Shape = TensorShape

    /// Element tensor shape
    public let elementShape: TensorShape?

    public var isScalar: Bool {
        return elementShape == nil
    }

    /// The number of units per element.
    public var unitCountPerElement: Int {
        return self.elementShape?.contiguousSize ?? 0
    }

    /// Contiguous storage
    public internal(set) var units: ContiguousArray<UnitType>

    /// Capacity reserved for subtensors
    public var capacity: Int {
        return units.capacity / unitCountPerElement
    }

    internal init(elementShape: TensorShape?, units: ContiguousArray<UnitType>) {
        let elementContiguousSize = elementShape?.contiguousSize ?? 1
        precondition(units.count % elementContiguousSize == 0,
                     "Unit count does not match element shape")
        self.elementShape = elementShape
        self.units = ContiguousArray(units)
    }
}

public extension Tensor {
    /// Initialize a tensor using an existing slice of elements in row-major order
    /// - parameter shape: tensor shape
    /// - parameter elements: slice of existing elements in row-major order
    internal init(shape: TensorShape, units: ContiguousArray<UnitType>) {
        precondition(units.count >= shape.contiguousSize,
                     "The slice has fewer elements than required by the shape")
        self.init(elementShape: shape.isScalar ? nil : shape.dropFirst(),
                  units: ContiguousArray(units.prefix(shape.contiguousSize)))
    }

    /// Initialize an empty tensor of scalar elements
    init() {
        self.init(elementShape: .scalar)
    }

    /// Initialize an empty tensor
    init(elementShape: TensorShape) {
        self.init(elementShape: elementShape, units: [])
    }

    /// Initialize a scalar tensor
    init(scalar: UnitType) {
        self.init(elementShape: nil, units: [scalar])
    }

    /// Initialize a tensor from a sequence of tensors of element shape
    init<S: Sequence>(elementShape: TensorShape, elements: S)
        where S.Element : TensorProtocol, S.Element.UnitType == UnitType {
            self.init(elementShape: elementShape)
            for element in elements {
                units.append(contentsOf: element.units)
            }
    }

    /// Initialize a tensor from a sequence of elements in row-major order
    /// - parameter shape: tensor shape
    /// - parameter elements: sequence of elements in row-major order
    /// - parameter vacancySupplier
    init<S: Sequence>(shape: TensorShape, units: S,
                      vacancySupplier supplier: (() -> UnitType)? = nil)
        where S.Iterator.Element == UnitType {
            let contiguousSize = shape.contiguousSize
            var slice = ContiguousArray(units.prefix(contiguousSize))
            /// If elements fewer than required by the shape and supplier is provided
            /// generate new elements using the supplier until vacancy is filled
            if slice.count < contiguousSize, let supplier = supplier {
                slice.reserveCapacity(shape.contiguousSize)
                slice.append(contentsOf: (0..<contiguousSize).map { _ in supplier() })
            }
            self.init(shape: shape, units: slice)
    }

    /// Allocate and initialize a tensor to a repeated value
    /// - parameter shape: tensor shape
    /// - parameter repeating: repeated value
    init(shape: TensorShape, repeating repeatedValue: UnitType) {
        let units = ContiguousArray(repeating: repeatedValue, count: shape.contiguousSize)
        self.init(shape: shape, units: units)
    }

    /// Allocate and initialize a tensor using the factory function
    /// - parameter shape: tensor shape
    /// - parameter supplier: factory function providing values lazily
    init(shape: TensorShape, supplier: () -> UnitType) {
        let units = ContiguousArray((0..<shape.contiguousSize).map { _ in supplier() })
        self.init(shape: shape, units: units)
    }

    /// Initialize a tensor from a TensorSlice
    init(_ slice: TensorSlice<UnitType>) {
        self.init(elementShape: slice.elementShape, units: ContiguousArray(slice.units))
    }
}

public extension Tensor where UnitType : Strideable {
    init(shape: TensorShape, unitsIncreasingFrom lowerBound: UnitType) {
        var unit = lowerBound
        self.init(shape: shape, supplier: {
            defer { unit = unit.advanced(by: 1) }
            return unit
        })
    }
}

public extension Tensor where UnitType : Strideable, UnitType.Stride : SignedInteger {
    init(scalarElementsIn bounds: CountableRange<UnitType>) {
        self.init(elementShape: .scalar, units: ContiguousArray(bounds))
    }

    init(scalarElementsIn bounds: CountableClosedRange<UnitType>) {
        self.init(elementShape: .scalar, units: ContiguousArray(bounds))
    }
}

public extension Tensor {
    func makeUnitIterator() -> IndexingIterator<ContiguousArray<UnitType>> {
        return units.makeIterator()
    }

    mutating func updateUnit(at index: Int, to newValue: UnitType) {
        units[units.startIndex.advanced(by: index)] = newValue
    }
}

public extension Tensor where UnitType : Numeric {
    mutating func incrementUnit(at index: Int, by newValue: UnitType) {
        units[units.startIndex.advanced(by: index)] += newValue
    }

    mutating func decrementUnit(at index: Int, by newValue: UnitType) {
        units[units.startIndex.advanced(by: index)] -= newValue
    }

    mutating func multiplyUnit(at index: Int, by newValue: UnitType) {
        units[units.startIndex.advanced(by: index)] *= newValue
    }
}

public extension Tensor where UnitType : BinaryInteger {
    mutating func divideUnit(at index: Int, by newValue: UnitType) {
        units[units.startIndex.advanced(by: index)] /= newValue
    }
}

public extension Tensor where UnitType : FloatingPoint {
    mutating func divideUnit(at index: Int, by newValue: UnitType) {
        units[units.startIndex.advanced(by: index)] /= newValue
    }
}

extension Tensor : RandomAccessCollection, RangeReplaceableCollection {
    public typealias Index = Int
    public typealias Element = TensorSlice<UnitType>
    public typealias SubSequence = TensorSlice<UnitType>

    /// Access the element tensor specified by a TensorIndex
    public subscript(index: TensorIndex) -> Element {
        get {
            precondition(!isScalar || index.isEmpty, "I am a scalar and I have no dimensions!")
            return TensorSlice(base: self, indices: index.elements)
        }
        set {
            precondition(!isScalar || index.isEmpty, "I am a scalar and I have no dimensions!")
            let newShape = shape.dropFirst(index.count)
            precondition(newShape == newValue.shape, "Shape mismatch")
            let contiguousIndex = index.contiguousIndex(in: shape)
            let range = contiguousIndex..<contiguousIndex+newShape.contiguousSize
            units.replaceSubrange(range, with: newValue.units)
        }
    }

    /// Access the element tensor specified by a list of dimensional indices
    /// - parameter indices: tensor indices
    /// - note: the count of indices must equal the raw rank of the tensor
    public subscript(indices: Int...) -> Element {
        get {
            return self[TensorIndex(indices)]
        }
        set {
            self[TensorIndex(indices)] = newValue
        }
    }

    /// Access the element tensor in the current dimension at an index
    public subscript(index: Int) -> Element {
        get {
            precondition(!isScalar, "I am a scalar and I have no dimensions!")
            return TensorSlice(base: self, index: index)
        }
        set {
            precondition(!isScalar, "I am a scalar and I have no dimensions!")
            let newShape = shape.dropFirst()
            precondition(newShape == newValue.shape, "Shape mismatch")
            let contiguousIndex = unitIndex(fromIndex: index)
            let range = contiguousIndex..<contiguousIndex+newShape.contiguousSize
            units.replaceSubrange(range, with: newValue.units)
        }
    }

    /// Access the subtensor specified by a contiguous range of indices
    public subscript(bounds: Range<Int>) -> SubSequence {
        get {
            precondition(!isScalar,
                         "I am a scalar and I have no dimensions!")
            precondition(indices ~= bounds.lowerBound && indices ~= bounds.upperBound - 1,
                         "Slice indices are out of bounds")
            return TensorSlice(base: self, bounds: CountableRange(bounds))
        }
        set {
            precondition(!isScalar,
                         "I am a scalar and I have no dimensions!")
            precondition(indices ~= bounds.lowerBound && indices ~= bounds.upperBound - 1,
                         "Slice indices are out of bounds")
            precondition(newValue.shape == elementShape?.prepending(bounds.count),
                         "Shape mismatch")
            units[unitSubrange(from: CountableRange(bounds))] =
                newValue.base.units[newValue.unitSubrange(from: newValue.indices)]
        }
    }

    public var count: Int {
        return isScalar ? 1 : units.count / unitCountPerElement
    }

    /// Returns a sequence of tensor indices for scalar elements
    public var indices: CountableRange<Int> {
        return 0..<count
    }

    public var startIndex: Int {
        return 0
    }

    public var endIndex: Int {
        return count
    }

    /// Returns the index after the specified one in the current dimension
    public func index(after i: Int) -> Int {
        return i + 1
    }

    /// Returns the index before the specified one in the current dimension
    public func index(before i: Int) -> Int {
        return i - 1
    }
}

public extension Tensor {
    func reshaped(as newShape: TensorShape) -> Tensor? {
        guard self.shape.contiguousSize == newShape.contiguousSize else {
            return nil
        }
        return Tensor(shape: newShape, units: units)
    }
}

public extension Tensor {
    func withUnsafeBufferPointer<Result>
        (_ body: (UnsafeBufferPointer<UnitType>) throws -> Result) rethrows -> Result {
        return try units.withUnsafeBufferPointer { ptr in
            try body(ptr)
        }
    }

    mutating func withUnsafeMutableBufferPointer<Result>
        (_ body: (inout UnsafeMutableBufferPointer<UnitType>) throws -> Result) rethrows -> Result {
        return try units.withUnsafeMutableBufferPointer { ptr in
            try body(&ptr)
        }
    }
}

extension Tensor : TextOutputStreamable {
    public func write<Target>(to target: inout Target) where Target : TextOutputStream {
        target.write(isScalar
            ? String(describing: units[0])
            : "[\(map {"\($0)"}.joined(separator: ", "))]")
    }
}
